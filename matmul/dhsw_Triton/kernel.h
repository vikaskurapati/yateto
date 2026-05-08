#ifndef YATETO_KERNEL_H_
#define YATETO_KERNEL_H_
#include <cmath>
#include <limits>
#include "yateto.h"
#include "tensor.h"
namespace yateto {
  namespace kernel {
    struct matmulAB {
      constexpr static unsigned long const NonZeroFlops = 64512;
      constexpr static unsigned long const HardwareFlops = 65536;
      constexpr static unsigned long const TmpMemRequiredInBytes = 0;
      constexpr static unsigned long const TmpMaxMemRequiredInBytes = 0;

      double const* A{};
      double const* B{};
      double* C{};


      void execute();
    };
  } // namespace kernel
  namespace kernel {
    struct matmulATB {
      constexpr static unsigned long const NonZeroFlops = 64512;
      constexpr static unsigned long const HardwareFlops = 65536;
      constexpr static unsigned long const TmpMemRequiredInBytes = 0;
      constexpr static unsigned long const TmpMaxMemRequiredInBytes = 0;

      double const* A{};
      double const* B{};
      double* C{};


      void execute();
    };
  } // namespace kernel
  namespace kernel {
    struct matmulABT {
      constexpr static unsigned long const NonZeroFlops = 64512;
      constexpr static unsigned long const HardwareFlops = 65536;
      constexpr static unsigned long const TmpMemRequiredInBytes = 0;
      constexpr static unsigned long const TmpMaxMemRequiredInBytes = 0;

      double const* A{};
      double const* B{};
      double* C{};


      void execute();
    };
  } // namespace kernel
  namespace kernel {
    struct matmulATBT {
      constexpr static unsigned long const NonZeroFlops = 64512;
      constexpr static unsigned long const HardwareFlops = 65536;
      constexpr static unsigned long const TmpMemRequiredInBytes = 8192;
      constexpr static unsigned long const TmpMaxMemRequiredInBytes = 8192;

      double const* A{};
      double const* B{};
      double* C{};


      void execute();
    };
  } // namespace kernel
} // namespace yateto
#endif
