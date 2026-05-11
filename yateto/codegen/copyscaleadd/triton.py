from ..common import BatchedOperationsAux
from ..cache import TritonWriter
from ..triton_common import TritonWrapper, TritonKernelArgument, TritonScalarKernelArgument
import hashlib

def _operand_parameters(name, address_mode):
  if address_mode == 'pointer_based':
    return [name, f'num_elements_{name}', f'extra_offset_{name}']
  if address_mode == 'strided':
    return [name, f'num_elements_{name}']
  if address_mode == 'none':
    return [name]
  raise ValueError(f'Unknown addressing mode: {address_mode}')

def _operand_base(name, address_mode, distance, floating_type):
  if address_mode == 'pointer_based':
    return f'tl.load({name} + batch_idx).to(tl.pointer_type({floating_type})) + extra_offset_{name}'
  if address_mode == 'strided':
    return f'{name} + batch_idx * {distance}'
  if address_mode == 'none':
    return name
  raise ValueError(f'Unknown addressing mode: {address_mode}')

class CopyScaleAddTriton(object):
    def __init__(self, arch, descr):
        self._arch = arch
        self._descr = descr

    def deduce_addressing(self, term):
        if term.is_compute_constant:
            return 'none'
        if term.is_temporary:
            return 'strided'
        else:
            return 'pointer_based'

    def generate(self, cpp, routineCache):
        d = self._descr
        
        indices = d.result.indices
        
        sizes = []
        starts = []
        for idx in indices:
            sizes.append(d.loopRanges[idx].size())
            starts.append(d.loopRanges[idx].start)
            
        unpack_code = []
        current_stride = 1
        for k, idx in enumerate(indices):
            size = sizes[k]
            start = starts[k]
            if k == 0:
                unpack_code.append(f"  {idx} = (pid % {size}) + {start}")
            else:
                unpack_code.append(f"  {idx} = ((pid // {current_stride}) % {size}) + {start}")
            current_stride *= size
            
        total_elements = current_stride
        
        term_addr = d.term.memoryLayout.addressString(d.term.indices, prefix='')
        if not term_addr:
            term_addr = "0"
            
        res_addr = d.result.memoryLayout.addressString(d.result.indices, prefix='')
        if not res_addr:
            res_addr = "0"

        alpha = d.alpha
        beta = d.beta
        
        address_mode_A = self.deduce_addressing(d.term)
        address_mode_B = self.deduce_addressing(d.result)
        
        params = []
        params.extend(_operand_parameters('A', address_mode_A))
        params.extend(_operand_parameters('B', address_mode_B))
        params.append('alpha')
        
        dist_A = d.term.memoryLayout.requiredReals()
        dist_B = d.result.memoryLayout.requiredReals()

        floating_type = 'tl.float64' if self._arch.bytesPerReal == 8 else 'tl.float32'

        base_A = _operand_base('A', address_mode_A, dist_A, floating_type)
        base_B = _operand_base('B', address_mode_B, dist_B, floating_type)
        
        kernel_source = f'''import triton
import triton.language as tl

@triton.jit
def copyscaleadd_kernel({", ".join(params)}):
  # In SeisSol, grid is 1D: number of batches.
  batch_idx = tl.program_id(0)
  
  # Inside the batch, we process all elements of the tensor sequentially using a loop 
  # or block. Here we process sequentially or vectorized for simplicity if N is small.
  # Triton works best with arange.
  # For CopyScaleAdd, elements are totally independent. We will use a loop if N is large,
  # or tl.arange if N is power of 2. For simplicity and robustness in AOT, we do:
  
  base_A = {base_A}
  base_B = {base_B}
  
  offs = tl.arange(0, {triton_next_power_of_2(total_elements)})
  mask = offs < {total_elements}
  
  pid = offs
{chr(10).join(unpack_code)}

  addr_A = base_A + {term_addr}
  addr_B = base_B + {res_addr}
  
  a_val = tl.load(addr_A, mask=mask)
'''
        if beta == 0.0:
            kernel_source += f'''  res = alpha * a_val
  tl.store(addr_B, res.to({floating_type}), mask=mask)
'''
        elif beta == 1.0:
            kernel_source += f'''  b_val = tl.load(addr_B, mask=mask)
  res = alpha * a_val + b_val
  tl.store(addr_B, res.to({floating_type}), mask=mask)
'''
        else:
            raise NotImplementedError(f"beta={beta} not supported")

        hash_str = hashlib.sha256(kernel_source.encode('utf-8')).hexdigest()[:16]
        kernel_name = f'csa_{hash_str}'
        kernel_source = kernel_source.replace('copyscaleadd_kernel', kernel_name)

        args = [
            TritonKernelArgument('A', d.term.name, d.term.is_compute_constant, d.term.is_temporary, False),
            TritonKernelArgument('B', d.result.name, d.result.is_compute_constant, d.result.is_temporary, True),
            TritonScalarKernelArgument('alpha', str(alpha))
        ]

        arch_name = self._arch.name if hasattr(self._arch, 'name') else 'sm_90a' # fallback
        
        wrapper = TritonWrapper(
            kernel_file=f"{kernel_name}.cubin",
            kernel_name=kernel_name,
            arguments=args,
            real_type=self._arch.typename,
            name=kernel_name + "_wrapper"
        )
        
        prototype = wrapper.prototype()
        routineCache.addRoutine(prototype, TritonWriter(prototype, wrapper, kernel_source, arch_name, wrapper.kernel_file))
        
        cpp(wrapper.call())
        
        flops = 1
        if d.beta != 0.0:
            flops += 1
        flops *= total_elements
        return flops

def triton_next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return max(16, n)
