#ifndef FASTEN_OPERATORS_UTILS
#define FASTEN_OPERATORS_UTILS

#include <torch/extension.h>

#include <map>

namespace fasten {

enum class Engine { TORCH = 0, MAGMA = 1, NATIVE = 2 };

typedef std::map<long, long> SliceIndex;

inline SliceIndex fasten_slice_index_build(const torch::TensorAccessor<long, 2> slice_accessor) {
  SliceIndex slice_index;
  for (auto i = 0; i < slice_accessor.size(0); ++i) {
    auto indicator = slice_accessor[i][0];
    slice_index[indicator] = i;
  }

  return slice_index;
}

}  // namespace fasten

#endif  // FASTEN_OPERATORS_UTILS