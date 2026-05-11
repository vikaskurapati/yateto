import unittest
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from yateto.codegen.gemm.triton import tritonGemmGen
from yateto.arch import HostArchDefinition, DeviceArchDefinition, deriveArchitecture

class TestTritonGemmGen(unittest.TestCase):
    def setUp(self):
        host_arch = HostArchDefinition('hsw', 'd', None, None)
        device_arch = DeviceArchDefinition('sm_90a', 'nvidia', 'cuda', 'd', None)
        self.arch = deriveArchitecture(host_arch, device_arch)
        
    def test_gemm_gen_basic(self):
        gd = {
            'M': 32,
            'N': 32,
            'K': 32,
            'LDA': 32,
            'LDB': 32,
            'LDC': 32,
            'addrA': 'pointer_based',
            'distA': 1024,
            'addrB': 'pointer_based',
            'distB': 1024,
            'addrC': 'pointer_based',
            'distC': 1024,
            'alpha': 1.0,
            'beta': 0.0,
            'transA': False,
            'transB': False,
        }
        
        kernel_source = tritonGemmGen(self.arch, gd)
        
        # Verify basic structure
        self.assertIn('import triton', kernel_source)
        self.assertIn('import triton.language as tl', kernel_source)
        self.assertIn('@triton.jit', kernel_source)
        
        # Check arguments and computation
        self.assertIn('def gemm_nn_k_32_m_32_n_32', kernel_source)
        self.assertIn('A, num_elements_A, extra_offset_A, B, num_elements_B, extra_offset_B, C, num_elements_C, extra_offset_C, alpha, beta', kernel_source)
        self.assertIn('acc = tl.zeros((32, 32), dtype=tl.float64)', kernel_source)
        self.assertIn('for kk in range(32):', kernel_source)
        self.assertIn('acc += a_vec[:, None] * b_vec[None, :]', kernel_source)
        
    def test_gemm_gen_transposed(self):
        gd = {
            'M': 16,
            'N': 16,
            'K': 16,
            'LDA': 16,
            'LDB': 16,
            'LDC': 16,
            'addrA': 'pointer_based',
            'distA': 256,
            'addrB': 'pointer_based',
            'distB': 256,
            'addrC': 'pointer_based',
            'distC': 256,
            'alpha': 1.0,
            'beta': 0.0,
            'transA': True,
            'transB': True,
        }
        
        kernel_source = tritonGemmGen(self.arch, gd)
        self.assertIn('gemm_tt', kernel_source)
        
        # Transpose logic check
        self.assertIn('a_vec = tl.load(base_A + kk + offs_m * 16)', kernel_source)
        self.assertIn('b_vec = tl.load(base_B + offs_n + kk * 16)', kernel_source)
        
    def test_gemm_gen_addressing_modes(self):
        gd = {
            'M': 8,
            'N': 8,
            'K': 8,
            'LDA': 8,
            'LDB': 8,
            'LDC': 8,
            'addrA': 'strided',
            'distA': 64,
            'addrB': 'none',
            'distB': 64,
            'addrC': 'strided',
            'distC': 64,
            'alpha': 2.0,
            'beta': 1.0,
            'transA': False,
            'transB': False,
        }
        
        kernel_source = tritonGemmGen(self.arch, gd)
        self.assertIn('A, num_elements_A, B, C, num_elements_C, alpha, beta', kernel_source)

    def test_gemm_gen_custom_kernel_name(self):
        gd = {
            'M': 12,
            'N': 9,
            'K': 7,
            'LDA': 16,
            'LDB': 9,
            'LDC': 12,
            'addrA': 'pointer_based',
            'distA': 108,
            'addrB': 'strided',
            'distB': 63,
            'addrC': 'pointer_based',
            'distC': 108,
            'alpha': 1.0,
            'beta': 0.0,
            'transA': False,
            'transB': True,
        }

        kernel_name = (
            'gemm_nn_addra_pointer_based_addrb_strided_addrc_pointer_based_'
            'alpha_1_0_beta_0_0_k_7_lda_16_ldb_9_ldc_12_m_12_n_9'
        )
        kernel_source = tritonGemmGen(self.arch, gd, kernel_name=kernel_name)

        self.assertIn(f'def {kernel_name}(', kernel_source)
        self.assertNotIn('def gemm_nt_', kernel_source)

if __name__ == '__main__':
    unittest.main()
