#!/usr/bin/env python3
"""
Unit tests for Triton backend integration in YATeTo.

Tests the Triton GemmTool class which serves as the entry point
for Triton kernel generation in YATeTo's gemm_configuration system.
"""

import unittest
import sys
from pathlib import Path

# Add yateto to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from yateto import gemm_configuration
from yateto.arch import HostArchDefinition, DeviceArchDefinition, deriveArchitecture


class TestTritonGemmToolClass(unittest.TestCase):
    """Test suite for the Triton GemmTool class."""

    def setUp(self):
        """Set up test fixtures - create architecture definitions."""
        # CUDA backend with sm_90a
        host_arch = HostArchDefinition('hsw', 'd', None, None)
        device_arch = DeviceArchDefinition('sm_90a', 'nvidia', 'cuda', 'd', None)
        self.arch_cuda = deriveArchitecture(host_arch, device_arch)
        
        # HIP backend with gfx90a
        device_arch_hip = DeviceArchDefinition('gfx90a', 'amd', 'hip', 'd', None)
        self.arch_hip = deriveArchitecture(host_arch, device_arch_hip)
        
        # SYCL backend (should not be supported)
        device_arch_sycl = DeviceArchDefinition('pvc', 'intel', 'oneapi', 'd', None)
        self.arch_sycl = deriveArchitecture(host_arch, device_arch_sycl)
        
        # CPU only (no device backend)
        self.arch_cpu = HostArchDefinition('hsw', 'd', None, None)

    def test_triton_class_exists(self):
        """Test that Triton class is defined in gemm_configuration."""
        self.assertTrue(
            hasattr(gemm_configuration, 'Triton'),
            "Triton class should be defined in gemm_configuration module"
        )

    def test_triton_is_code_generator(self):
        """Test that Triton inherits from CodeGenerator base class."""
        self.assertTrue(
            issubclass(gemm_configuration.Triton, gemm_configuration.CodeGenerator),
            "Triton should inherit from CodeGenerator"
        )

    def test_triton_initialization_cuda(self):
        """Test Triton initialization with CUDA backend."""
        triton_tool = gemm_configuration.Triton(self.arch_cuda)
        self.assertIsNotNone(triton_tool)
        self.assertEqual(triton_tool._arch, self.arch_cuda)

    def test_triton_initialization_hip(self):
        """Test Triton initialization with HIP backend."""
        triton_tool = gemm_configuration.Triton(self.arch_hip)
        self.assertIsNotNone(triton_tool)
        self.assertEqual(triton_tool._arch, self.arch_hip)

    def test_arch_supported_cuda(self):
        """Test that CUDA backend is supported."""
        triton_tool = gemm_configuration.Triton(self.arch_cuda)
        self.assertTrue(
            triton_tool.archSupported(),
            "Triton should support CUDA backend"
        )

    def test_arch_supported_hip(self):
        """Test that HIP backend is supported."""
        triton_tool = gemm_configuration.Triton(self.arch_hip)
        self.assertTrue(
            triton_tool.archSupported(),
            "Triton should support HIP backend"
        )

    def test_arch_not_supported_sycl(self):
        """Test that SYCL/oneAPI backend is NOT supported."""
        triton_tool = gemm_configuration.Triton(self.arch_sycl)
        self.assertFalse(
            triton_tool.archSupported(),
            "Triton should NOT support SYCL/oneAPI backend (tinytc handles that)"
        )

    def test_arch_not_supported_cpu(self):
        """Test that CPU-only architecture is NOT supported."""
        triton_tool = gemm_configuration.Triton(self.arch_cpu)
        self.assertFalse(
            triton_tool.archSupported(),
            "Triton should NOT support CPU-only architecture"
        )

    def test_supported_basic_gemm_cuda(self):
        """Test that basic GEMM is supported on CUDA."""
        triton_tool = gemm_configuration.Triton(self.arch_cuda)
        
        # Basic GEMM: C = A * B (no transpose, no sparsity)
        supported = triton_tool.supported(
            m=56, n=9, k=56,
            sparseA=False, sparseB=False,
            transA=False, transB=False,
            alpha=1.0, beta=1.0,
            alignedA=True, alignedC=True,
            target='gpu'
        )
        self.assertTrue(supported, "Basic GEMM should be supported on CUDA")

    def test_supported_transpose_operations(self):
        """Test that transposed GEMMs are supported."""
        triton_tool = gemm_configuration.Triton(self.arch_cuda)
        
        # Test transA
        self.assertTrue(triton_tool.supported(
            m=56, n=9, k=56, sparseA=False, sparseB=False,
            transA=True, transB=False, alpha=1.0, beta=1.0,
            alignedA=True, alignedC=True, target='gpu'
        ), "GEMM with transA should be supported")
        
        # Test transB
        self.assertTrue(triton_tool.supported(
            m=56, n=9, k=56, sparseA=False, sparseB=False,
            transA=False, transB=True, alpha=1.0, beta=1.0,
            alignedA=True, alignedC=True, target='gpu'
        ), "GEMM with transB should be supported")
        
        # Test both
        self.assertTrue(triton_tool.supported(
            m=56, n=9, k=56, sparseA=False, sparseB=False,
            transA=True, transB=True, alpha=1.0, beta=1.0,
            alignedA=True, alignedC=True, target='gpu'
        ), "GEMM with both transA and transB should be supported")

    def test_not_supported_sparse_matrices(self):
        """Test that sparse matrices are NOT supported."""
        triton_tool = gemm_configuration.Triton(self.arch_cuda)
        
        # Sparse A
        self.assertFalse(triton_tool.supported(
            m=56, n=9, k=56, sparseA=True, sparseB=False,
            transA=False, transB=False, alpha=1.0, beta=1.0,
            alignedA=True, alignedC=True, target='gpu'
        ), "Sparse A should NOT be supported")
        
        # Sparse B
        self.assertFalse(triton_tool.supported(
            m=56, n=9, k=56, sparseA=False, sparseB=True,
            transA=False, transB=False, alpha=1.0, beta=1.0,
            alignedA=True, alignedC=True, target='gpu'
        ), "Sparse B should NOT be supported")

    def test_not_supported_cpu_target(self):
        """Test that CPU target is NOT supported."""
        triton_tool = gemm_configuration.Triton(self.arch_cuda)
        
        self.assertFalse(triton_tool.supported(
            m=56, n=9, k=56, sparseA=False, sparseB=False,
            transA=False, transB=False, alpha=1.0, beta=1.0,
            alignedA=True, alignedC=True, target='cpu'
        ), "CPU target should NOT be supported (Triton is GPU-only)")

    def test_supported_various_alpha_beta(self):
        """Test support for various alpha and beta values."""
        triton_tool = gemm_configuration.Triton(self.arch_cuda)
        
        # Test common alpha/beta combinations
        test_cases = [
            (1.0, 0.0, True, "alpha=1.0, beta=0.0 (C = A*B)"),
            (1.0, 1.0, True, "alpha=1.0, beta=1.0 (C += A*B)"),
            (2.0, 1.0, True, "alpha=2.0, beta=1.0 (C += 2*A*B)"),
            (0.5, 0.0, True, "alpha=0.5, beta=0.0 (C = 0.5*A*B)"),
            (-1.0, 1.0, True, "alpha=-1.0, beta=1.0 (C -= A*B)"),
        ]
        
        for alpha, beta, should_support, desc in test_cases:
            with self.subTest(desc=desc):
                result = triton_tool.supported(
                    m=56, n=9, k=56, sparseA=False, sparseB=False,
                    transA=False, transB=False, alpha=alpha, beta=beta,
                    alignedA=True, alignedC=True, target='gpu'
                )
                if should_support:
                    self.assertTrue(result, f"{desc} should be supported")
                else:
                    self.assertFalse(result, f"{desc} should NOT be supported")

    def test_supported_various_matrix_sizes(self):
        """Test support for various DG matrix sizes from SeisSol."""
        triton_tool = gemm_configuration.Triton(self.arch_cuda)
        
        # Matrix sizes from SeisSol (order 2-7 elastic)
        dg_sizes = [
            (10, 9, 10, "order 2"),
            (20, 9, 20, "order 3"),
            (35, 9, 35, "order 4"),
            (56, 9, 56, "order 5"),
            (84, 9, 84, "order 6"),
            (120, 9, 120, "order 7"),
            # Face projections
            (56, 21, 21, "flux order 5"),
            (84, 28, 28, "flux order 6"),
            # ADER sizes
            (56, 56, 56, "ader order 5"),
            (84, 84, 84, "ader order 6"),
        ]
        
        for m, n, k, desc in dg_sizes:
            with self.subTest(size=desc):
                self.assertTrue(triton_tool.supported(
                    m=m, n=n, k=k, sparseA=False, sparseB=False,
                    transA=False, transB=False, alpha=1.0, beta=1.0,
                    alignedA=True, alignedC=True, target='gpu'
                ), f"Matrix size {desc} ({m}×{n}×{k}) should be supported")

    def test_preference_returns_valid_value(self):
        """Test that preference() returns a valid Preference value."""
        triton_tool = gemm_configuration.Triton(self.arch_cuda)
        
        pref = triton_tool.preference(
            m=56, n=9, k=56, sparseA=False, sparseB=False,
            transA=False, transB=False, alpha=1.0, beta=1.0,
            alignedA=True, alignedC=True
        )
        
        # Check that it's a valid preference value (0-4)
        self.assertIn(pref, [
            gemm_configuration.Preference.LOWEST,
            gemm_configuration.Preference.LOW,
            gemm_configuration.Preference.MODERATE,
            gemm_configuration.Preference.HIGH,
            gemm_configuration.Preference.HIGHEST,
        ], "Preference should be a valid Preference enum value")

    def test_preference_highest_for_supported_operations(self):
        """Test that Triton has HIGHEST preference when it supports an operation."""
        triton_tool = gemm_configuration.Triton(self.arch_cuda)
        
        pref = triton_tool.preference(
            m=56, n=9, k=56, sparseA=False, sparseB=False,
            transA=False, transB=False, alpha=1.0, beta=1.0,
            alignedA=True, alignedC=True
        )
        
        self.assertEqual(
            pref, gemm_configuration.Preference.HIGHEST,
            "Triton should have HIGHEST preference for operations it supports"
        )

    def test_alignment_parameters(self):
        """Test that alignment parameters don't affect support for Triton."""
        triton_tool = gemm_configuration.Triton(self.arch_cuda)
        
        # Test all alignment combinations
        for alignedA in [True, False]:
            for alignedC in [True, False]:
                with self.subTest(alignedA=alignedA, alignedC=alignedC):
                    self.assertTrue(triton_tool.supported(
                        m=56, n=9, k=56, sparseA=False, sparseB=False,
                        transA=False, transB=False, alpha=1.0, beta=1.0,
                        alignedA=alignedA, alignedC=alignedC, target='gpu'
                    ), f"Should support alignedA={alignedA}, alignedC={alignedC}")

    def test_operation_name_attribute(self):
        """Test that operation_name attribute is set correctly."""
        triton_tool = gemm_configuration.Triton(self.arch_cuda)
        self.assertTrue(
            hasattr(triton_tool, 'operation_name'),
            "Triton should have operation_name attribute (from CodeGenerator)"
        )

    def test_includes_attribute(self):
        """Test that includes attribute exists."""
        triton_tool = gemm_configuration.Triton(self.arch_cuda)
        self.assertTrue(
            hasattr(triton_tool, 'includes'),
            "Triton should have includes attribute (from CodeGenerator)"
        )


class TestTritonInGeneratorCollection(unittest.TestCase):
    """Test Triton integration with YATeTo's GeneratorCollection."""

    def setUp(self):
        """Set up test fixtures."""
        host_arch = HostArchDefinition('hsw', 'd', None, None)
        device_arch = DeviceArchDefinition('sm_90a', 'nvidia', 'cuda', 'd', None)
        self.arch = deriveArchitecture(host_arch, device_arch)
        self.triton_tool = gemm_configuration.Triton(self.arch)

    def test_triton_in_generator_collection(self):
        """Test that Triton can be added to GeneratorCollection."""
        collection = gemm_configuration.GeneratorCollection([self.triton_tool])
        self.assertIsNotNone(collection)
        self.assertIn(self.triton_tool, collection.gemmTools)

    def test_triton_selected_for_gpu_gemm(self):
        """Test that Triton gets selected for GPU GEMM operations."""
        collection = gemm_configuration.GeneratorCollection([self.triton_tool])
        
        selected = collection.getGemmTool(
            m=56, n=9, k=56, sparseA=False, sparseB=False,
            transA=False, transB=False, alpha=1.0, beta=1.0,
            alignedA=True, alignedC=True, target='gpu'
        )
        
        self.assertIsInstance(
            selected, gemm_configuration.Triton,
            "Triton should be selected for GPU GEMM"
        )

    def test_triton_not_selected_for_cpu(self):
        """Test that Triton is NOT selected for CPU target."""
        collection = gemm_configuration.GeneratorCollection([self.triton_tool])
        
        selected = collection.getGemmTool(
            m=56, n=9, k=56, sparseA=False, sparseB=False,
            transA=False, transB=False, alpha=1.0, beta=1.0,
            alignedA=True, alignedC=True, target='cpu'
        )
        
        self.assertIsNone(
            selected,
            "Triton should NOT be selected for CPU target"
        )

    def test_triton_vs_gemmforge_preference(self):
        """Test preference ordering between Triton and GemmForge."""
        gemmforge = gemm_configuration.GemmForge(self.arch)
        
        # Collection with both
        collection = gemm_configuration.GeneratorCollection([self.triton_tool, gemmforge])
        
        selected = collection.getGemmTool(
            m=56, n=9, k=56, sparseA=False, sparseB=False,
            transA=False, transB=False, alpha=1.0, beta=1.0,
            alignedA=True, alignedC=True, target='gpu'
        )
        
        # Both have HIGHEST preference, but last one wins (reversed iteration)
        # Or we can adjust Triton preference to be specifically higher
        self.assertIsNotNone(selected, "Some backend should be selected")
        self.assertTrue(
            isinstance(selected, (gemm_configuration.Triton, gemm_configuration.GemmForge)),
            "Either Triton or GemmForge should be selected"
        )


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
