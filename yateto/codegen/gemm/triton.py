from __future__ import annotations

from ..triton_common import make_triton_kernel_name


def _operand_parameters(name, address_mode):
  if address_mode == 'pointer_based':
    return [name, f'num_elements_{name}', f'extra_offset_{name}'], f'num_elements_{name}'
  if address_mode == 'strided':
    return [name, f'num_elements_{name}'], f'num_elements_{name}'
  if address_mode == 'none':
    return [name], None
  raise ValueError(f'Unknown addressing mode: {address_mode}')


def _operand_base(name, address_mode, distance):
  if address_mode == 'pointer_based':
    return f'tl.load({name} + pid) + extra_offset_{name}'
  if address_mode == 'strided':
    return f'{name} + pid * {distance}'
  if address_mode == 'none':
    return name
  raise ValueError(f'Unknown addressing mode: {address_mode}')


def tritonGemmGen(arch, gd, kernel_name=''):
  if not kernel_name:
    kernel_name = make_triton_kernel_name('gemm',
                                          transpose_a=gd['transA'],
                                          transpose_b=gd['transB'],
                                          m=gd['M'],
                                          n=gd['N'],
                                          k=gd['K'])

  params = []
  batch_limit = None
  base_statements = []
  for op in ['A', 'B', 'C']:
    address_mode = gd[f'addr{op}']
    op_params, num_elements = _operand_parameters(op, address_mode)
    params.extend(op_params)
    if batch_limit is None and num_elements is not None:
      batch_limit = num_elements
    base_statements.append(f'  base_{op} = {_operand_base(op, address_mode, gd[f"dist{op}"])}')

  params.extend(['alpha', 'beta'])

  if gd['transA']:
    ptr_a = f'  a_ptrs = base_A + offs_k[:, None] + offs_m[None, :] * {gd["LDA"]}'
    load_a = '  a = tl.trans(tl.load(a_ptrs))'
  else:
    ptr_a = f'  a_ptrs = base_A + offs_m[:, None] + offs_k[None, :] * {gd["LDA"]}'
    load_a = '  a = tl.load(a_ptrs)'

  if gd['transB']:
    ptr_b = f'  b_ptrs = base_B + offs_n[None, :] + offs_k[:, None] * {gd["LDB"]}'
  else:
    ptr_b = f'  b_ptrs = base_B + offs_k[:, None] + offs_n[None, :] * {gd["LDB"]}'

  batch_guard = ''
  if batch_limit is not None:
    batch_guard = f'  if pid >= {batch_limit}:\n    return\n'

  floating_type = 'tl.float64' if arch.bytesPerReal == 8 else 'tl.float32'
  return f"""import triton
import triton.language as tl

@triton.jit
def {kernel_name}({', '.join(params)}):
  pid = tl.program_id(0)
{batch_guard}{chr(10).join(base_statements)}
  offs_m = tl.arange(0, {gd['M']})
  offs_n = tl.arange(0, {gd['N']})
  offs_k = tl.arange(0, {gd['K']})

{ptr_a}
{load_a}
{ptr_b}
  b = tl.load(b_ptrs)

  acc = tl.dot(a, b).to({floating_type})
  c_ptrs = base_C + offs_m[:, None] + offs_n[None, :] * {gd['LDC']}
  c = alpha * acc + beta * tl.load(c_ptrs)
  tl.store(c_ptrs, c)
"""
