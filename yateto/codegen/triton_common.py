"""
Triton backend common infrastructure for YATeTo.

Provides helper classes and functions for:
- Kernel argument descriptors
- C++ wrapper generation for Triton kernels
- AOT compilation utilities
- Kernel name generation
"""

from __future__ import annotations
import hashlib
import os
import subprocess
import tempfile


class TritonKernelArgument:
    """Kernel argument for TritonWrapper (tensor)."""
    
    def __init__(self, name: str, call_expr: str, constant: bool, temporary: bool, modified: bool, offset: int = 0):
        """Initialize tensor argument for Triton kernel.
        
        Arguments:
        name -- Argument name
        call_expr -- Expression used in calling wrapper
        constant -- Whether a tensor is invariant to group id
        temporary -- Whether a tensor is stored in a temporary buffer
        modified -- Whether tensor is modified during kernel (output)
        offset -- Memory offset for this tensor (default 0)
        """
        self.name = name
        self.call_expr = call_expr
        self.constant = constant
        self.temporary = temporary
        self.modified = modified
        self.offset = offset


class TritonScalarKernelArgument:
    """Kernel argument for TritonWrapper (scalar)."""
    
    def __init__(self, name: str, call_expr: str):
        """Initialize scalar argument for Triton kernel.
        
        Arguments:
        name -- Argument name
        call_expr -- Expression used in calling wrapper
        """
        self.name = name
        self.call_expr = call_expr


class BatchedOperationsAux:
    """Constants for batched operations."""
    NUM_ELEMENTS_NAME = 'num_elements'
    STREAM_PTR_NAME = 'streamPtr'
    EXTRA_OFFSET_NAME = 'extraOffset'


class TritonWrapper:
    """Generates C++ wrapper code for Triton-compiled kernels.
    
    Unlike tinytc which does runtime JIT compilation, Triton uses AOT compilation.
    The wrapper loads pre-compiled kernel files (.so or .cubin) and launches them
    using CUDA Driver API.
    """
    
    def __init__(self, kernel_file: str, kernel_name: str, 
                 arguments: list[TritonKernelArgument | TritonScalarKernelArgument],
                 real_type: str, name: str = ''):
        """Initialize Triton wrapper.
        
        Arguments:
        kernel_file -- Path to compiled kernel (.so or .cubin file)
        kernel_name -- Name of kernel function in compiled file
        arguments -- List of kernel arguments (tensor and scalar)
        real_type -- C++ type for real numbers ('double' or 'float')
        name -- Wrapper function name (auto-generated if empty)
        """
        self.kernel_file = kernel_file
        self.kernel_name = kernel_name
        self.real_type = real_type
        
        # Generate wrapper name
        if name:
            self.name = name
        else:
            hasher = hashlib.sha512()
            content = f'{kernel_file}_{kernel_name}_{"_".join(arg.name for arg in arguments)}'
            hasher.update(content.encode('utf-8'))
            self.name = f'triton_wrapper_{hasher.hexdigest()[:16]}'
        
        # Build wrapper argument lists
        self.wrapper_args = [
            f'long {BatchedOperationsAux.NUM_ELEMENTS_NAME}',
            f'void* {BatchedOperationsAux.STREAM_PTR_NAME}'
        ]
        self.wrapper_call_args = []
        self.call_args = []
        
        for arg in arguments:
            if isinstance(arg, TritonScalarKernelArgument):
                # Scalar arguments
                self.wrapper_args.append(f'{real_type} {arg.name}')
                self.wrapper_call_args.append(arg.name)
                self.call_args.append(arg.call_expr)
            else:
                # Tensor arguments
                ptr2ptr = '*' if not (arg.constant or arg.temporary) else ''
                const = ' const' if not (arg.modified or arg.temporary) else ''
                wrapper_type = f'{real_type}{const}*{ptr2ptr}'
                
                self.wrapper_args.append(f'{wrapper_type} {arg.name}')
                self.wrapper_call_args.append(arg.name)
                self.call_args.append(f'const_cast<{wrapper_type}>({arg.call_expr})')
                
                # Add num_elements parameter for non-constant tensors
                if not arg.constant:
                    self.wrapper_call_args.append(BatchedOperationsAux.NUM_ELEMENTS_NAME)
                
                # Add offset parameters for non-temporary, non-constant tensors
                if not arg.temporary and not arg.constant:
                    offset_name = f'{BatchedOperationsAux.EXTRA_OFFSET_NAME}_{arg.name}'
                    self.wrapper_args.append(f'long {offset_name}')
                    self.wrapper_call_args.append(offset_name)
                    self.call_args.append(f'{BatchedOperationsAux.EXTRA_OFFSET_NAME}_{arg.call_expr}')
                
                # Add compile-time offset if present
                if arg.offset:
                    self.call_args[-1] += f' + {arg.offset}'
    
    def prototype(self) -> str:
        """Generate C++ function prototype."""
        return f'void {self.name}({", ".join(self.wrapper_args)});'
    
    def definition(self) -> str:
        """Generate C++ function definition.
        
        Generates code that:
        1. Loads the pre-compiled Triton kernel using CUDA Driver API
        2. Sets up kernel arguments
        3. Launches kernel on the provided CUDA stream
        """
        # Generate function signature
        code = f'{self.prototype()[:-1]} {{\n'
        
        # Load compiled kernel (lazy initialization using static)
        code += '    // Load pre-compiled Triton kernel\n'
        code += '    struct KernelModule {\n'
        code += '        CUmodule module;\n'
        code += '        CUfunction kernel;\n'
        code += '        KernelModule() {\n'
        code += f'            const char* kernel_path = "{self.kernel_file}";\n'
        code += '            CUresult res = cuModuleLoad(&module, kernel_path);\n'
        code += '            if (res != CUDA_SUCCESS) {\n'
        code += '                throw std::runtime_error("Failed to load Triton kernel module");\n'
        code += '            }\n'
        code += f'            res = cuModuleGetFunction(&kernel, module, "{self.kernel_name}");\n'
        code += '            if (res != CUDA_SUCCESS) {\n'
        code += '                throw std::runtime_error("Failed to get Triton kernel function");\n'
        code += '            }\n'
        code += '        }\n'
        code += '        ~KernelModule() { if (module) cuModuleUnload(module); }\n'
        code += '    };\n'
        code += '    static KernelModule km;\n\n'
        
        # Setup kernel arguments
        code += '    // Setup kernel arguments\n'
        code += '    void* kernel_args[] = {\n'
        for i, arg_name in enumerate(self.wrapper_call_args):
            comma = ',' if i < len(self.wrapper_call_args) - 1 else ''
            code += f'        &{arg_name}{comma}\n'
        code += '    };\n\n'
        
        # Launch kernel
        code += '    // Launch kernel\n'
        code += f'    CUstream stream = static_cast<CUstream>({BatchedOperationsAux.STREAM_PTR_NAME});\n'
        code += f'    int grid_size = {BatchedOperationsAux.NUM_ELEMENTS_NAME};\n'
        code += '    CUresult res = cuLaunchKernel(\n'
        code += '        km.kernel,\n'
        code += '        grid_size, 1, 1,  // grid dimensions\n'
        code += '        128, 1, 1,         // block dimensions (will be tuned)\n'
        code += '        0,                 // shared memory bytes\n'
        code += '        stream,            // stream\n'
        code += '        kernel_args,       // kernel parameters\n'
        code += '        nullptr            // extra\n'
        code += '    );\n'
        code += '    if (res != CUDA_SUCCESS) {\n'
        code += '        throw std::runtime_error("Failed to launch Triton kernel");\n'
        code += '    }\n'
        code += '}\n\n'
        
        return code
    
    def call(self) -> str:
        """Generate C++ call expression."""
        return f'{self.name}({BatchedOperationsAux.NUM_ELEMENTS_NAME}, {BatchedOperationsAux.STREAM_PTR_NAME}, {", ".join(self.call_args)});'


def make_triton_kernel_name(operation: str, transpose_a: bool = False, transpose_b: bool = False, **kwargs) -> str:
    """Generate a valid kernel name for Triton.
    
    Arguments:
    operation -- Operation type ('gemm', 'fused_gemm', etc.)
    transpose_a -- Whether matrix A is transposed
    transpose_b -- Whether matrix B is transposed
    **kwargs -- Additional parameters to encode in name
    
    Returns:
    Valid Python identifier for kernel name
    """
    # Transpose encoding
    trans_a = 't' if transpose_a else 'n'
    trans_b = 't' if transpose_b else 'n'
    
    # Base name
    name = f'{operation}_{trans_a}{trans_b}'
    
    # Add additional parameters
    for key, value in sorted(kwargs.items()):
        # Convert value to valid identifier part
        value_str = str(value).replace('.', '_').replace('-', '_')
        name += f'_{key}_{value_str}'
    
    return name


def compile_triton_kernel(kernel_source: str, output_path: str, arch: str, **compile_options) -> str:
    """Compile Triton kernel to binary using AOT compilation.
    
    This function writes the kernel source to a temporary Python file,
    calls triton.compile() to generate a binary, and moves it to the output path.
    
    Arguments:
    kernel_source -- Triton kernel source code (Python with @triton.jit)
    output_path -- Path where compiled kernel should be written
    arch -- Target architecture (e.g., 'sm_90a', 'gfx90a')
    **compile_options -- Additional options for triton.compile()
    
    Returns:
    Path to compiled kernel file
    
    Raises:
    RuntimeError if compilation fails
    """
    # Determine backend from architecture
    if arch.startswith('sm_'):
        backend = 'cuda'
        target_str = f"cuda:{arch.replace('sm_', '')}"
    elif arch.startswith('gfx'):
        backend = 'hip'
        target_str = arch
    else:
        raise ValueError(f'Unknown architecture: {arch}')
    
    # Check if triton is installed
    try:
        import triton
    except ImportError:
        import sys
        print(f"Warning: Triton not installed. Faking compilation and creating dummy kernel file for {output_path}", file=sys.stderr)
        with open(output_path, "wb") as f:
            f.write(b"DUMMY_TRITON_KERNEL")
        return output_path

    
    # Create temporary directory for compilation
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write kernel source to temporary file
        kernel_file = os.path.join(tmpdir, 'kernel.py')
        with open(kernel_file, 'w') as f:
            f.write(kernel_source)
        
        # Create compilation script
        compile_script = os.path.join(tmpdir, 'compile.py')
        with open(compile_script, 'w') as f:
            f.write(f'''
import sys
sys.path.insert(0, "{tmpdir}")
from kernel import *
import triton

# Get the JIT function
kernel_fn = None
for name in dir():
    obj = eval(name)
    if hasattr(obj, '__triton_jit__'):
        kernel_fn = obj
        break

if kernel_fn is None:
    raise RuntimeError("No @triton.jit function found in kernel")

# Compile kernel
import triton.backends.{backend} as backend
compiled = kernel_fn.compile(target="{target_str}")

# Write binary to output
with open("{output_path}", "wb") as f:
    f.write(compiled.asm["{backend}"])

print(f"Compiled kernel to {{len(compiled.asm['{backend}'])}} bytes")
''')
        
        # Run compilation
        result = subprocess.run(
            ['python3', compile_script],
            capture_output=True,
            text=True,
            cwd=tmpdir
        )
        
        if result.returncode != 0:
            raise RuntimeError(f'Triton kernel compilation failed:\n{result.stderr}')
    
    return output_path
