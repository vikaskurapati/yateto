import os

from .code import Cpp
from .triton_common import compile_triton_kernel

class RoutineGenerator(object):
  def __call__(self, routineName, fileName):
    pass

  def target(self):
    return 'cpu'

class GpuRoutineGenerator(object):
  def __call__(self, routineName, fileName):
    pass

  def target(self):
    return 'gpu'

class RoutineCache(object):
  def __init__(self):
    self._routines = dict()
    self._generators = dict()

  def addRoutine(self, name, generator):
    if name in self._routines and not self._routines[name] == generator:
      raise RuntimeError(f'`{name}` is already in RoutineCache but the generator is not equal. '
                         f'(That is, a name was given twice for different routines.)')
    self._routines[name] = generator

    generatorName = type(generator).__name__
    if generatorName not in self._generators:
      self._generators[generatorName] = generator

  def generate(self, header, cppFileName, gpuFileName):
    with Cpp(gpuFileName) as gpucpp:
      with Cpp(cppFileName) as cpp:
        for generator in self._generators.values():
          if generator.target() == 'gpu':
            generator.header(gpucpp)
          elif generator.target() == 'cpu':
            generator.header(cpp)
          else:
            raise NotImplementedError(f'Unknown target: {generator.target()}')

    for name, generator in self._routines.items():
      if generator.target() == 'gpu':
        declaration = generator(name, gpuFileName)
      elif generator.target() == 'cpu':
        declaration = generator(name, cppFileName)
      else:
        raise NotImplementedError(f'Unknown target: {generator.target()}')
      header(declaration)

class TinytcWriter(GpuRoutineGenerator):
  def __init__(self, signature, source):
    self._source = source
    self._signature = signature

  def __eq__(self, other):
    return self._signature == other._signature

  def header(self, cpp):
    cpp.include('tinytc/tinytc.hpp')
    cpp.include('tinytc/tinytc_sycl.hpp')
    cpp.includeSys('sycl/sycl.hpp')
    cpp.includeSys('stdexcept')
    cpp.includeSys('utility')

  def __call__(self, routineName, fileName):
    with open(fileName, 'a') as f:
      f.write(self._source)

    return self._signature

class TritonWriter(GpuRoutineGenerator):
  def __init__(self, signature, wrapper, kernel_source, arch, kernel_file):
    self._signature = signature
    self._wrapper = wrapper
    self._kernel_source = kernel_source
    self._arch = arch
    self._kernel_file = kernel_file

  def __eq__(self, other):
    return self._signature == other._signature

  def header(self, cpp):
    cpp.includeSys('cuda.h')
    cpp.includeSys('stdexcept')

  def __call__(self, routineName, fileName):
    output_dir = os.path.dirname(fileName)
    kernel_path = os.path.join(output_dir, self._kernel_file)
    compile_triton_kernel(self._kernel_source, kernel_path, self._arch)

    self._wrapper.kernel_file = kernel_path
    with open(fileName, 'a') as f:
      f.write(self._wrapper.definition())

    return self._signature
