#ifndef YATETO_INIT_H_
#define YATETO_INIT_H_
#include "tensor.h"
#include "yateto.h"
namespace yateto {
  namespace init {
    struct A : tensor::A {
      constexpr static unsigned const Start[] = {0, 0};
      constexpr static unsigned const Stop[] = {32, 32};

      struct view {
        using type = ::yateto::DenseTensorView<2,double,unsigned,false>;
        using type_const = ::yateto::DenseTensorView<2,double,unsigned,true>;
        static inline type create(double * values) {
          return ::yateto::DenseTensorView<2,double,unsigned,false>(values, {32, 32}, {0, 0}, {32, 32});
        }
        static inline type_const create(double const* values) {
          return ::yateto::DenseTensorView<2,double,unsigned,true>(values, {32, 32}, {0, 0}, {32, 32});
        }
      };
    };
    struct B : tensor::B {
      constexpr static unsigned const Start[] = {0, 0};
      constexpr static unsigned const Stop[] = {32, 32};

      struct view {
        using type = ::yateto::DenseTensorView<2,double,unsigned,false>;
        using type_const = ::yateto::DenseTensorView<2,double,unsigned,true>;
        static inline type create(double * values) {
          return ::yateto::DenseTensorView<2,double,unsigned,false>(values, {32, 32}, {0, 0}, {32, 32});
        }
        static inline type_const create(double const* values) {
          return ::yateto::DenseTensorView<2,double,unsigned,true>(values, {32, 32}, {0, 0}, {32, 32});
        }
      };
    };
    struct C : tensor::C {
      constexpr static unsigned const Start[] = {0, 0};
      constexpr static unsigned const Stop[] = {32, 32};

      struct view {
        using type = ::yateto::DenseTensorView<2,double,unsigned,false>;
        using type_const = ::yateto::DenseTensorView<2,double,unsigned,true>;
        static inline type create(double * values) {
          return ::yateto::DenseTensorView<2,double,unsigned,false>(values, {32, 32}, {0, 0}, {32, 32});
        }
        static inline type_const create(double const* values) {
          return ::yateto::DenseTensorView<2,double,unsigned,true>(values, {32, 32}, {0, 0}, {32, 32});
        }
      };
    };
  } // namespace init
  namespace init {
  } // namespace init
} // namespace yateto
#endif
