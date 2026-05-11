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


def _operand_base(name, address_mode, distance, float_type):
  if address_mode == 'pointer_based':
    return f'tl.load({name} + pid).to(tl.pointer_type({float_type})) + extra_offset_{name}'
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
  floating_type = 'tl.float64' if arch.bytesPerReal == 8 else 'tl.float32'
  params = []
  batch_limit = None
  base_statements = []
  for op in ['A', 'B', 'C']:
    address_mode = gd[f'addr{op}']
    op_params, num_elements = _operand_parameters(op, address_mode)
    params.extend(op_params)
    if batch_limit is None and num_elements is not None:
      batch_limit = num_elements
    base_statements.append(f'  base_{op} = {_operand_base(op, address_mode, gd[f"dist{op}"], floating_type)}')

  params.extend(['alpha', 'beta'])

  if gd['transA']:
    a_index = f'kk + i * {gd["LDA"]}'
  else:
    a_index = f'i + kk * {gd["LDA"]}'

  if gd['transB']:
    b_index = f'j + kk * {gd["LDB"]}'
  else:
    b_index = f'kk + j * {gd["LDB"]}'

  batch_guard = ''
  if batch_limit is not None:
    batch_guard = f'  if pid >= {batch_limit}:\n    return\n'

  return f"""import triton
import triton.language as tl

@triton.jit
def {kernel_name}({', '.join(params)}):
  pid = tl.program_id(0)
{batch_guard}{chr(10).join(base_statements)}
  for i in tl.static_range(0, {gd['M']}):
    for j in tl.static_range(0, {gd['N']}):
      acc = tl.full((), 0.0, {floating_type})
      for kk in tl.static_range(0, {gd['K']}):
        a_val = tl.load(base_A + {a_index})
        b_val = tl.load(base_B + {b_index})
        acc += a_val * b_val
      c_ptr = base_C + i + j * {gd['LDC']}
      c_old = tl.load(c_ptr)
      c_val = alpha * acc + beta * c_old
      tl.store(c_ptr, c_val)
"""
