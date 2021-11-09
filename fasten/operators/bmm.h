#ifndef FASTEN_OPERATORS_BMM
#define FASTEN_OPERATORS_BMM

#include <torch/extension.h>

namespace fasten {

enum class BmmEngine { TORCH = 1, MAGMA = 2, NATIVE = 3 };

template <BmmEngine engine>
void bmm_forward(torch::Tensor input,
                 torch::TensorAccessor<size_t, 2> input_slices_accessor,
                 torch::Tensor weight,
                 torch::TensorAccessor<size_t, 2> weight_slices_accessor,
                 torch::Tensor output);

template <BmmEngine engine>
void bmm_backward(torch::Tensor grad, torch::Tensor input,
                  torch::TensorAccessor<size_t, 2> input_slices_accessor,
                  torch::Tensor weight,
                  torch::TensorAccessor<size_t, 2> weight_slices_accessor,
                  torch::Tensor input_grad, torch::Tensor weight_grad);

}  // namespace fasten

#endif  // FASTEN_OPERATORS_BMM