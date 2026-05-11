from __future__ import annotations

from ..triton_common import make_triton_kernel_name


def _next_power_of_two(value):
  power_of_two = 1
  while power_of_two < value:
    power_of_two *= 2
  return power_of_two


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
    a_index = f'kk + offs_m * {gd["LDA"]}'
  else:
    a_index = f'offs_m + kk * {gd["LDA"]}'

  if gd['transB']:
    b_index = f'offs_n + kk * {gd["LDB"]}'
  else:
    b_index = f'kk + offs_n * {gd["LDB"]}'

  batch_guard = ''
  if batch_limit is not None:
    batch_guard = f'  if pid >= {batch_limit}:\n    return\n'

  floating_type = 'tl.float64' if arch.bytesPerReal == 8 else 'tl.float32'
  tile_m = _next_power_of_two(gd['M'])
  tile_n = _next_power_of_two(gd['N'])
  return f"""import triton
import triton.language as tl

@triton.jit
def {kernel_name}({', '.join(params)}):
  pid = tl.program_id(0)
{batch_guard}{chr(10).join(base_statements)}
  offs_m = tl.arange(0, {tile_m})
  offs_n = tl.arange(0, {tile_n})
  mask_m = offs_m < {gd['M']}
  mask_n = offs_n < {gd['N']}

  acc = tl.zeros(({tile_m}, {tile_n}), dtype={floating_type})
  for kk in tl.static_range(0, {gd['K']}):
    a_vec = tl.load(base_A + {a_index}, mask=mask_m, other=0.0)
    b_vec = tl.load(base_B + {b_index}, mask=mask_n, other=0.0)
    acc += a_vec[:, None] * b_vec[None, :]

  c_ptrs = base_C + offs_m[:, None] + offs_n[None, :] * {gd['LDC']}
  c_mask = mask_m[:, None] & mask_n[None, :]
  c_old = tl.load(c_ptrs, mask=c_mask, other=0.0)
  c_val = alpha * acc + beta * c_old
  tl.store(c_ptrs, c_val, mask=c_mask)
"""
