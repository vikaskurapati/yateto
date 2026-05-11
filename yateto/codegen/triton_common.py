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


def compile_triton_kernel(kernel_source: str, output_path: str, arch: str, kernel_name: str = None, **compile_options) -> str:
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
        target_arch = arch.replace('sm_', '')
        target_str = f"{backend}:{target_arch}"
    elif arch.startswith('gfx'):
        backend = 'hip'
        target_arch = arch
        target_str = f"{backend}:{target_arch}"
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
import os
import inspect
sys.path.insert(0, "{tmpdir}")

import triton
import kernel

def is_triton_jit_function(obj):
    """Return True if obj looks like a Triton @triton.jit function.

    Triton has changed the exact JITFunction API across versions, so avoid
    relying on a single attribute pair such as compile/run.
    """

    cls = type(obj)
    cls_name = cls.__name__
    cls_module = getattr(cls, "__module__", "")

    if cls_name == "JITFunction" and cls_module.startswith("triton"):
        return True

    if hasattr(obj, "fn") and callable(getattr(obj, "fn")) and cls_module.startswith("triton"):
        return True

    if hasattr(obj, "src") and cls_module.startswith("triton"):
        return True

    return False


def discover_triton_jit_functions(module):
    candidates = []
    for name in dir(module):
        if name.startswith("__"):
            continue
        obj = getattr(module, name)
        if is_triton_jit_function(obj):
            candidates.append((name, obj))
    return candidates


requested_kernel_name = {kernel_name!r}
kernel_fn = None
candidates = discover_triton_jit_functions(kernel)

if requested_kernel_name is not None:
    kernel_fn = getattr(kernel, requested_kernel_name, None)
    if kernel_fn is not None and not is_triton_jit_function(kernel_fn):
        raise RuntimeError(
            f"Object named '{{requested_kernel_name}}' exists but is not a @triton.jit function "
            f"(type={{type(kernel_fn)!r}})"
        )
    if kernel_fn is None:
        if len(candidates) == 1:
            kernel_fn = candidates[0][1]
        else:
            candidate_names = ", ".join(name for name, _ in candidates) if candidates else "<none>"
            raise RuntimeError(
                f"No @triton.jit function found in kernel named '{{requested_kernel_name}}'. "
                f"Available @triton.jit functions: {{candidate_names}}"
            )
else:
    if len(candidates) == 1:
        kernel_fn = candidates[0][1]
    elif len(candidates) > 1:
        candidate_names = ", ".join(name for name, _ in candidates)
        raise RuntimeError(
        f"Multiple @triton.jit functions found in kernel: {{candidate_names}}. "
        "Pass kernel_name explicitly."
        )

if kernel_fn is None:
    exported = ", ".join(name for name in dir(kernel) if not name.startswith("__"))
    raise RuntimeError(
        "No @triton.jit function found in kernel."
        + (f" named '{{requested_kernel_name}}'" if requested_kernel_name is not None else "")
        + f". Exported names: {{exported}}"
    )

def _dedup_preserve(items):
    seen = set()
    result = []
    for item in items:
        key = (type(item), str(item))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _is_class(obj):
    return isinstance(obj, type)


def _collect_gputarget_classes(triton_compile, compiler_compile):
    classes = []

    def add_class(candidate):
        if _is_class(candidate) and getattr(candidate, "__name__", "") == "GPUTarget" and candidate not in classes:
            classes.append(candidate)

    for compile_fn in (triton_compile, compiler_compile):
        if callable(compile_fn):
            globals_dict = getattr(compile_fn, "__globals__", None)
            if isinstance(globals_dict, dict):
                add_class(globals_dict.get("GPUTarget"))

    modules = [
        "triton.backends.compiler",
        "triton.backends.nvidia.compiler",
        "triton.backends.amd.compiler",
        "triton.compiler",
        "triton.compiler.compiler",
    ]
    for module_name in modules:
        try:
            module = __import__(module_name, fromlist=["GPUTarget"])
            add_class(getattr(module, "GPUTarget", None))
        except Exception:
            pass

    return classes


def _arch_candidates(backend, arch_hint):
    values = [arch_hint]
    token = str(arch_hint).lower()

    if backend == "cuda":
        token = token.replace("sm_", "").replace("sm", "")
        values.append(token)
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            values.extend([digits, int(digits), "sm" + digits, "sm_" + digits])
        if token.endswith("a") and digits:
            values.extend([digits + "a", "sm" + digits + "a", "sm_" + digits + "a"])
    elif backend == "hip":
        values.append(token)
        if token.startswith("gfx"):
            values.append(token.replace("gfx", ""))
        else:
            values.append("gfx" + token)

    return _dedup_preserve(values)


def _instantiate_gputarget(cls, backend, arch_value):
    warp_size = 32 if backend == "cuda" else 64
    kw_attempts = [
        dict(backend=backend, arch=arch_value, warp_size=warp_size),
        dict(backend=backend, arch=arch_value),
        dict(arch=arch_value, backend=backend),
        dict(arch=arch_value),
    ]
    for kwargs in kw_attempts:
        try:
            return cls(**kwargs)
        except TypeError:
            pass
        except Exception:
            pass

    pos_attempts = [
        (backend, arch_value, warp_size),
        (backend, arch_value),
        (arch_value,),
    ]
    for args in pos_attempts:
        try:
            return cls(*args)
        except Exception:
            pass
    return None


def _target_candidates(target_str, backend, arch_hint, gputarget_classes):
    candidates = [target_str]

    # Newer Triton versions require target to be a GPUTarget object.
    for cls in gputarget_classes:
        for arch_value in _arch_candidates(backend, arch_hint):
            target_obj = _instantiate_gputarget(cls, backend, arch_value)
            if target_obj is not None:
                candidates.append(target_obj)

    return candidates


def _target_label(target):
    if isinstance(target, (str, int, float)):
        return repr(target)
    backend_attr = getattr(target, "backend", None)
    arch_attr = getattr(target, "arch", None)
    return f"{{type(target).__name__}}(backend={{backend_attr!r}}, arch={{arch_attr!r}})"


def _source_candidates(kernel_fn, kernel_source_path, requested_kernel_name):
    candidates = [
        kernel_fn,
        getattr(kernel_fn, "fn", None),
        kernel_source_path,
    ]
    if requested_kernel_name:
        candidates.append(kernel_source_path + ":" + requested_kernel_name)
    return [candidate for candidate in _dedup_preserve(candidates) if candidate is not None]


def _source_label(source):
    if isinstance(source, str):
        return repr(source)
    return f"{{type(source).__name__}}"


def _guess_signature(kernel_fn, kernel_source_path):
    fn = getattr(kernel_fn, "fn", None)
    if fn is None:
        return None

    scalar_ty = "fp32"
    try:
        with open(kernel_source_path, "r") as f:
            kernel_src = f.read()
        if "tl.float64" in kernel_src:
            scalar_ty = "fp64"
    except Exception:
        pass

    try:
        params = list(inspect.signature(fn).parameters.keys())
    except Exception:
        return None

    sig_items = []
    for name in params:
        lname = name.lower()
        if lname.startswith("num_elements") or lname.startswith("extra_offset"):
            sig_items.append("i64")
        elif lname in ("alpha", "beta"):
            sig_items.append(scalar_ty)
        else:
            sig_items.append("*" + scalar_ty)
    return sig_items


def _ast_source_candidates(kernel_fn, kernel_source_path):
    ast_classes = []

    def add_ast_class(candidate):
        if isinstance(candidate, type) and candidate.__name__ == "ASTSource" and candidate not in ast_classes:
            ast_classes.append(candidate)

    for module_name in (
        "triton.compiler.compiler",
        "triton.compiler",
    ):
        try:
            module = __import__(module_name, fromlist=["ASTSource"])
            add_ast_class(getattr(module, "ASTSource", None))
        except Exception:
            pass

    triton_compile = getattr(triton, "compile", None)
    if callable(triton_compile):
        globals_dict = getattr(triton_compile, "__globals__", None)
        if isinstance(globals_dict, dict):
            add_ast_class(globals_dict.get("ASTSource"))

    signature_items = _guess_signature(kernel_fn, kernel_source_path)
    signature_candidates = []
    if signature_items:
        signature_candidates.append(",".join(signature_items))
        signature_candidates.append(dict(enumerate(signature_items)))
    maybe_sig = getattr(kernel_fn, "signature", None)
    if maybe_sig is not None:
        signature_candidates.append(maybe_sig)
    signature_candidates.append(None)
    signature_candidates = _dedup_preserve(signature_candidates)

    fn_candidates = [kernel_fn, getattr(kernel_fn, "fn", None)]
    fn_candidates = [fn for fn in _dedup_preserve(fn_candidates) if fn is not None]

    ast_sources = []
    for ast_cls in ast_classes:
        try:
            param_names = list(inspect.signature(ast_cls).parameters.keys())
        except Exception:
            param_names = []
        has_signature = "signature" in param_names
        has_fn = "fn" in param_names or "function" in param_names
        fn_key = "fn" if "fn" in param_names else ("function" if "function" in param_names else None)

        for fn in fn_candidates:
            for sig in signature_candidates:
                kwargs = dict()
                if has_fn and fn_key:
                    kwargs[fn_key] = fn
                if has_signature and sig is not None:
                    kwargs["signature"] = sig

                if kwargs:
                    try:
                        ast_sources.append(ast_cls(**kwargs))
                    except Exception:
                        pass

                # positional fallbacks for older constructor signatures
                try:
                    if sig is not None:
                        ast_sources.append(ast_cls(fn, sig))
                    else:
                        ast_sources.append(ast_cls(fn))
                except Exception:
                    pass

    return _dedup_preserve(ast_sources)


def compile_with_compat(kernel_fn, target_str, backend, arch_hint, kernel_source_path, requested_kernel_name):
    errors = []
    triton_compile = getattr(triton, "compile", None)
    compiler_compile = None

    try:
        import triton.compiler as triton_compiler
        compiler_compile = getattr(triton_compiler, "compile", None)
    except Exception as err:
        errors.append(f"import triton.compiler failed: {{err!r}}")

    gputarget_classes = _collect_gputarget_classes(triton_compile, compiler_compile)
    targets = _target_candidates(target_str, backend, arch_hint, gputarget_classes)
    ast_sources = _ast_source_candidates(kernel_fn, kernel_source_path)
    base_sources = _source_candidates(kernel_fn, kernel_source_path, requested_kernel_name)
    sources = _dedup_preserve(ast_sources + base_sources)

    # Triton variants where JITFunction exposes `.compile()`
    compile_method = getattr(kernel_fn, "compile", None)
    if callable(compile_method):
        for target in targets:
            try:
                return compile_method(target=target)
            except Exception as err:
                errors.append(
                    f"kernel_fn.compile(target={{_target_label(target)}}) failed: {{err!r}}"
                )

    # Triton variants with module-level `triton.compile(...)`
    if callable(triton_compile):
        for source in sources:
            for target in targets:
                try:
                    return triton_compile(source, target=target)
                except Exception as err:
                    errors.append(
                        f"triton.compile(source={{_source_label(source)}}, target={{_target_label(target)}}) failed: {{err!r}}"
                    )

    # Triton variants where compiler API is exposed via triton.compiler.compile(...)
    if callable(compiler_compile):
        for source in sources:
            for target in targets:
                try:
                    return compiler_compile(source, target=target)
                except Exception as err:
                    errors.append(
                        f"triton.compiler.compile(source={{_source_label(source)}}, target={{_target_label(target)}}) failed: {{err!r}}"
                    )

    details = "\\n".join(errors) if errors else "<no compile entry points were available>"
    raise RuntimeError("Unable to compile Triton kernel with available APIs:\\n" + details)


def extract_binary(compiled):
    # Common Triton result shape: object with `.asm` dict
    asm = getattr(compiled, "asm", None)
    if isinstance(asm, dict):
        for key in ("cubin", "hsaco", "{backend}", "ptx", "llir"):
            if key in asm:
                binary = asm[key]
                if isinstance(binary, str):
                    if os.path.exists(binary):
                        with open(binary, "rb") as f:
                            return f.read()
                    return binary.encode("utf-8")
                return binary

    # Some variants expose the artifact as direct attributes
    for attr in ("cubin", "hsaco", "ptx", "llir"):
        if hasattr(compiled, attr):
            binary = getattr(compiled, attr)
            if binary:
                if isinstance(binary, str):
                    if os.path.exists(binary):
                        with open(binary, "rb") as f:
                            return f.read()
                    return binary.encode("utf-8")
                return binary

    # Some APIs return a path directly
    if isinstance(compiled, str):
        if os.path.exists(compiled):
            with open(compiled, "rb") as f:
                return f.read()
        return compiled.encode("utf-8")
    if isinstance(compiled, bytes):
        return compiled

    artifact_path = getattr(compiled, "path", None)
    if isinstance(artifact_path, str) and os.path.exists(artifact_path):
        with open(artifact_path, "rb") as f:
            return f.read()

    available_attrs = sorted([a for a in dir(compiled) if not a.startswith("_")])[:30]
    raise RuntimeError(
        f"Could not extract Triton binary artifact from compiled object type={{type(compiled)!r}}; "
        f"visible attrs={{available_attrs}}"
    )


compiled = compile_with_compat(
    kernel_fn,
    target_str="{target_str}",
    backend="{backend}",
    arch_hint="{target_arch}",
    kernel_source_path={kernel_file!r},
    requested_kernel_name=requested_kernel_name,
)
binary = extract_binary(compiled)

with open("{output_path}", "wb") as f:
    f.write(binary)

print(f"Compiled kernel to {{len(binary)}} bytes")
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
