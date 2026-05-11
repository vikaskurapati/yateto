#!/usr/bin/env python3

import unittest
import sys
import os

# Add yateto to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from yateto.memory import DenseMemoryLayout


class TestTritonKernelArgument(unittest.TestCase):
    """Test TritonKernelArgument class"""
    
    def test_class_exists(self):
        """TritonKernelArgument class should exist in triton_common"""
        from yateto.codegen.triton_common import TritonKernelArgument
        self.assertTrue(TritonKernelArgument is not None)
    
    def test_init_tensor_argument(self):
        """Should initialize tensor argument with all required fields"""
        from yateto.codegen.triton_common import TritonKernelArgument
        
        arg = TritonKernelArgument(
            name='A',
            call_expr='A_ptr',
            constant=False,
            temporary=False,
            modified=False,
            offset=0
        )
        
        self.assertEqual(arg.name, 'A')
        self.assertEqual(arg.call_expr, 'A_ptr')
        self.assertFalse(arg.constant)
        self.assertFalse(arg.temporary)
        self.assertFalse(arg.modified)
        self.assertEqual(arg.offset, 0)
    
    def test_init_with_offset(self):
        """Should support non-zero offset"""
        from yateto.codegen.triton_common import TritonKernelArgument
        
        arg = TritonKernelArgument(
            name='B',
            call_expr='B_ptr',
            constant=True,
            temporary=False,
            modified=False,
            offset=128
        )
        
        self.assertEqual(arg.offset, 128)
        self.assertTrue(arg.constant)
    
    def test_modified_argument(self):
        """Should handle modified (output) tensors"""
        from yateto.codegen.triton_common import TritonKernelArgument
        
        arg = TritonKernelArgument(
            name='C',
            call_expr='C_ptr',
            constant=False,
            temporary=False,
            modified=True,
            offset=0
        )
        
        self.assertTrue(arg.modified)
        self.assertFalse(arg.constant)


class TestTritonScalarKernelArgument(unittest.TestCase):
    """Test TritonScalarKernelArgument class"""
    
    def test_class_exists(self):
        """TritonScalarKernelArgument class should exist"""
        from yateto.codegen.triton_common import TritonScalarKernelArgument
        self.assertTrue(TritonScalarKernelArgument is not None)
    
    def test_init_scalar(self):
        """Should initialize scalar argument"""
        from yateto.codegen.triton_common import TritonScalarKernelArgument
        
        arg = TritonScalarKernelArgument(name='alpha', call_expr='1.0')
        
        self.assertEqual(arg.name, 'alpha')
        self.assertEqual(arg.call_expr, '1.0')


class TestTritonWrapper(unittest.TestCase):
    """Test TritonWrapper class for C++ wrapper generation"""
    
    def setUp(self):
        from yateto.codegen.triton_common import TritonKernelArgument, TritonScalarKernelArgument
        
        self.tensor_args = [
            TritonKernelArgument('A', 'A_ptr', False, False, False, 0),
            TritonKernelArgument('B', 'B_ptr', True, False, False, 0),
            TritonKernelArgument('C', 'C_ptr', False, False, True, 0)
        ]
        
        self.scalar_args = [
            TritonScalarKernelArgument('alpha', '1.0'),
            TritonScalarKernelArgument('beta', '0.0')
        ]
    
    def test_class_exists(self):
        """TritonWrapper class should exist"""
        from yateto.codegen.triton_common import TritonWrapper
        self.assertTrue(TritonWrapper is not None)
    
    def test_init_basic(self):
        """Should initialize wrapper with kernel info"""
        from yateto.codegen.triton_common import TritonWrapper
        
        wrapper = TritonWrapper(
            kernel_file='gemm_kernel.so',
            kernel_name='gemm_nn',
            arguments=self.tensor_args + self.scalar_args,
            real_type='double',
            name='my_gemm_wrapper'
        )
        
        self.assertEqual(wrapper.kernel_file, 'gemm_kernel.so')
        self.assertEqual(wrapper.kernel_name, 'gemm_nn')
        self.assertEqual(wrapper.name, 'my_gemm_wrapper')
        self.assertEqual(wrapper.real_type, 'double')
    
    def test_init_auto_name(self):
        """Should generate hash-based name if not provided"""
        from yateto.codegen.triton_common import TritonWrapper
        
        wrapper = TritonWrapper(
            kernel_file='gemm_kernel.so',
            kernel_name='gemm_nn',
            arguments=self.tensor_args,
            real_type='double'
        )
        
        # Should have generated a name
        self.assertTrue(wrapper.name.startswith('triton_wrapper_'))
        self.assertGreater(len(wrapper.name), len('triton_wrapper_'))
    
    def test_prototype_generation(self):
        """Should generate C++ function prototype"""
        from yateto.codegen.triton_common import TritonWrapper
        
        wrapper = TritonWrapper(
            kernel_file='gemm_kernel.so',
            kernel_name='gemm_nn',
            arguments=self.tensor_args + self.scalar_args,
            real_type='double',
            name='test_wrapper'
        )
        
        proto = wrapper.prototype()
        
        # Should contain function signature
        self.assertIn('void test_wrapper(', proto)
        self.assertIn('long num_elements', proto)
        self.assertIn('void* streamPtr', proto)
        # Should end with semicolon
        self.assertTrue(proto.strip().endswith(';'))
    
    def test_definition_generation(self):
        """Should generate C++ function definition"""
        from yateto.codegen.triton_common import TritonWrapper
        
        wrapper = TritonWrapper(
            kernel_file='gemm_kernel.so',
            kernel_name='gemm_nn',
            arguments=self.tensor_args,
            real_type='double',
            name='test_wrapper'
        )
        
        defn = wrapper.definition()
        
        # Should contain function definition
        self.assertIn('void test_wrapper(', defn)
        self.assertIn('{', defn)
        self.assertIn('}', defn)
        # Should reference kernel file
        self.assertIn('gemm_kernel.so', defn)
        # Should call kernel launch
        self.assertIn('gemm_nn', defn)
    
    def test_call_generation(self):
        """Should generate C++ call expression"""
        from yateto.codegen.triton_common import TritonWrapper
        
        wrapper = TritonWrapper(
            kernel_file='gemm_kernel.so',
            kernel_name='gemm_nn',
            arguments=self.tensor_args + self.scalar_args,
            real_type='double',
            name='test_wrapper'
        )
        
        call = wrapper.call()
        
        # Should contain wrapper name and call
        self.assertIn('test_wrapper(', call)
        self.assertIn('num_elements', call)
        self.assertIn('streamPtr', call)
        # Should end with semicolon
        self.assertTrue(call.strip().endswith(';'))
    
    def test_handles_const_tensors(self):
        """Should handle constant tensors correctly"""
        from yateto.codegen.triton_common import TritonKernelArgument, TritonWrapper
        
        const_arg = TritonKernelArgument('B', 'B_ptr', True, False, False, 0)
        
        wrapper = TritonWrapper(
            kernel_file='test.so',
            kernel_name='test_kernel',
            arguments=[const_arg],
            real_type='double'
        )
        
        proto = wrapper.prototype()
        # Constant tensors should be handled differently (fewer parameters)
        self.assertIn('B', proto)
    
    def test_handles_temporary_tensors(self):
        """Should handle temporary tensors correctly"""
        from yateto.codegen.triton_common import TritonKernelArgument, TritonWrapper
        
        temp_arg = TritonKernelArgument('tmp', 'tmp_ptr', False, True, False, 0)
        
        wrapper = TritonWrapper(
            kernel_file='test.so',
            kernel_name='test_kernel',
            arguments=[temp_arg],
            real_type='double'
        )
        
        proto = wrapper.prototype()
        # Temporary tensors have different handling
        self.assertIn('tmp', proto)


class TestTritonCompiler(unittest.TestCase):
    """Test Triton AOT compilation helpers"""
    
    def test_compile_kernel_function_exists(self):
        """compile_triton_kernel function should exist"""
        from yateto.codegen.triton_common import compile_triton_kernel
        self.assertTrue(callable(compile_triton_kernel))
    
    def test_compile_kernel_signature(self):
        """compile_triton_kernel should accept kernel source and config"""
        from yateto.codegen.triton_common import compile_triton_kernel
        import inspect
        
        sig = inspect.signature(compile_triton_kernel)
        params = list(sig.parameters.keys())
        
        # Should have at minimum: kernel source, output path, architecture
        self.assertIn('kernel_source', params)
        self.assertIn('output_path', params)
        self.assertIn('arch', params)

    def test_compile_kernel_contains_api_compat_fallbacks(self):
        """compile_triton_kernel should support multiple Triton compile APIs"""
        from yateto.codegen.triton_common import compile_triton_kernel
        import inspect

        src = inspect.getsource(compile_triton_kernel)
        self.assertIn('kernel_fn.compile', src)
        self.assertIn('triton.compile(', src)
        self.assertIn('triton.compiler', src)
        self.assertIn('GPUTarget', src)


class TestTritonHelpers(unittest.TestCase):
    """Test helper functions for Triton code generation"""
    
    def test_make_triton_kernel_name(self):
        """Should generate valid kernel names"""
        from yateto.codegen.triton_common import make_triton_kernel_name
        
        name = make_triton_kernel_name('gemm', transpose_a=False, transpose_b=False)
        
        # Should be valid identifier
        self.assertTrue(name.isidentifier())
        # Should contain operation type
        self.assertIn('gemm', name.lower())
    
    def test_make_triton_kernel_name_transpose(self):
        """Should encode transpose information in name"""
        from yateto.codegen.triton_common import make_triton_kernel_name
        
        name_nn = make_triton_kernel_name('gemm', transpose_a=False, transpose_b=False)
        name_nt = make_triton_kernel_name('gemm', transpose_a=False, transpose_b=True)
        name_tn = make_triton_kernel_name('gemm', transpose_a=True, transpose_b=False)
        name_tt = make_triton_kernel_name('gemm', transpose_a=True, transpose_b=True)
        
        # All should be different
        names = {name_nn, name_nt, name_tn, name_tt}
        self.assertEqual(len(names), 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
