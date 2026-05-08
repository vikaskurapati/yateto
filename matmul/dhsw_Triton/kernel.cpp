#include <cassert>
#include <cstring>
#include <cstdlib>
#include <limits>
#include "subroutine.h"
#include "kernel.h"
namespace yateto {
  constexpr unsigned long const kernel::matmulAB::NonZeroFlops;
  constexpr unsigned long const kernel::matmulAB::HardwareFlops;
  void kernel::matmulAB::execute() {
    assert(A != nullptr);
    assert(B != nullptr);
    assert(C != nullptr);
    for (int n = 0; n < 32; ++n) {
      for (int m = 0; m < 32; ++m) {
        C[0 + 1*m + 32*n] = 0.0;
      }
      for (int k = 0; k < 32; ++k) {
        for (int m = 0; m < 32; ++m) {
          C[0 + 1*m + 32*n] += 1.0 * A[0 + 1*m + 32*k] * B[0 + 1*k + 32*n];
        }
      }
    }
  }
  constexpr unsigned long const kernel::matmulATB::NonZeroFlops;
  constexpr unsigned long const kernel::matmulATB::HardwareFlops;
  void kernel::matmulATB::execute() {
    assert(A != nullptr);
    assert(B != nullptr);
    assert(C != nullptr);
    for (int n = 0; n < 32; ++n) {
      for (int m = 0; m < 32; ++m) {
        C[0 + 1*m + 32*n] = 0.0;
      }
      for (int k = 0; k < 32; ++k) {
        for (int m = 0; m < 32; ++m) {
          C[0 + 1*m + 32*n] += 1.0 * A[0 + 32*m + 1*k] * B[0 + 1*k + 32*n];
        }
      }
    }
  }
  constexpr unsigned long const kernel::matmulABT::NonZeroFlops;
  constexpr unsigned long const kernel::matmulABT::HardwareFlops;
  void kernel::matmulABT::execute() {
    assert(A != nullptr);
    assert(B != nullptr);
    assert(C != nullptr);
    for (int n = 0; n < 32; ++n) {
      for (int m = 0; m < 32; ++m) {
        C[0 + 1*m + 32*n] = 0.0;
      }
      for (int k = 0; k < 32; ++k) {
        for (int m = 0; m < 32; ++m) {
          C[0 + 1*m + 32*n] += 1.0 * A[0 + 1*m + 32*k] * B[0 + 32*k + 1*n];
        }
      }
    }
  }
  constexpr unsigned long const kernel::matmulATBT::NonZeroFlops;
  constexpr unsigned long const kernel::matmulATBT::HardwareFlops;
  void kernel::matmulATBT::execute() {
    assert(A != nullptr);
    assert(B != nullptr);
    assert(C != nullptr);
    double *_tmp0;
    alignas(32) double _buffer0[1024] ;
    _tmp0 = _buffer0;
    for (int n = 0; n < 32; ++n) {
      for (int m = 0; m < 32; ++m) {
        _tmp0[0 + 1*m + 32*n] = 0.0;
      }
      for (int k = 0; k < 32; ++k) {
        for (int m = 0; m < 32; ++m) {
          _tmp0[0 + 1*m + 32*n] += 1.0 * B[0 + 1*m + 32*k] * A[0 + 1*k + 32*n];
        }
      }
    }
    for (int _j = 0; _j < 32; ++_j) {
      #pragma omp simd
      for (int _i = 0; _i < 32; ++_i) {
        C[1*_i + 32*_j] = _tmp0[1*_j + 32*_i];
      }
    }
  }
} // namespace yateto
