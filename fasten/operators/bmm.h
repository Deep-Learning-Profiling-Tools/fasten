#ifndef FASTEN_OPERATORS_BMM
#define FASTEN_OPERATORS_BMM

#include <torch/extension.h>

#include "utils.h"

namespace fasten {

template <Engine engine>
void bmm_forward(torch::Tensor input, torch::TensorAccessor<size_t, 2> input_slices_accessor,
                 torch::Tensor weight, torch::TensorAccessor<size_t, 2> weight_slices_accessor,
                 torch::Tensor output);

template <Engine engine>
void bmm_backward(torch::Tensor grad, torch::Tensor input,
                  torch::TensorAccessor<size_t, 2> input_slices_accessor, torch::Tensor weight,
                  torch::TensorAccessor<size_t, 2> weight_slices_accessor, torch::Tensor input_grad,
                  torch::Tensor weight_grad);

}  // namespace fasten

#endif  // FASTEN_OPERATORS_BMM