#ifndef FASTEN_OPERATORS_UTILS
#define FASTEN_OPERATORS_UTILS

#include <torch/extension.h>

#include <map>

namespace fasten {

typedef std::map<size_t, size_t> SliceIndex;

inline SliceIndex fasten_slice_index_build(const torch::TensorAccessor<size_t, 2> slice_accessor) {
  SliceIndex slice_index;
  for (size_t i = 0; i < slice_accessor.size(0); ++i) {
    size_t indicator = slice_accessor[i][0];
    slice_index[indicator] = i;
  }

  return slice_index;
}

}  // namespace fasten

#endif  // FASTEN_OPERATORS_UTILS