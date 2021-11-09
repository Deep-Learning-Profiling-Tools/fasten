#ifndef FASTEN_OPERATORS_DETAILS_BMM_TORCH
#define FASTEN_OPERATORS_DETAILS_BMM_TORCH

#include <torch/extension.h>

#include "../bmm.h"

namespace fasten {

template <>
void bmm_forward<BmmEngine::TORCH>(torch::Tensor input,
                                   torch::TensorAccessor<size_t, 2> input_slices_accessor,
                                   torch::Tensor weight,
                                   torch::TensorAccessor<size_t, 2> weight_slices_accessor,
                                   torch::Tensor output) {
  std::map<size_t, size_t> weight_slices_index;
  for (size_t i = 0; i < weight_slices_accessor.size(0); ++i) {
    size_t indicator = weight_slices_accessor[i][0];
    weight_slices_index[indicator] = i;
  }

  using torch::indexing::Slice;
  for (size_t i = 0; i < input_slices_accessor.size(0); ++i) {
    // TODO(Keren): weight index check
    auto indicator = input_slices_accessor[i][0];
    auto weight_index = weight_slices_index.at(indicator);
    auto input_start = input_slices_accessor[i][1];
    auto input_end = input_slices_accessor[i][2];
    auto weight_start = weight_slices_accessor[weight_index][1];
    auto weight_end = weight_slices_accessor[weight_index][2];
    auto sub_input_tensor = input.index({Slice(0, input_start, input_end)});
    auto sub_weight_tensor = weight.index({Slice(0, weight_start, weight_end)});
    auto sub_output_tensor = output.index({Slice(0, input_start, input_end)});
    sub_output_tensor = torch::mm(sub_input_tensor, sub_weight_tensor);
  }
}

template <>
void bmm_backward<BmmEngine::TORCH>(torch::Tensor grad, torch::Tensor input,
                                    torch::TensorAccessor<size_t, 2> input_slices_accessor,
                                    torch::Tensor weight,
                                    torch::TensorAccessor<size_t, 2> weight_slices_accessor,
                                    torch::Tensor input_grad, torch::Tensor weight_grad) {
  using torch::indexing::Slice;

  std::map<size_t, size_t> weight_slices_index;
  for (size_t i = 0; i < weight_slices_accessor.size(0); ++i) {
    size_t indicator = weight_slices_accessor[i][0];
    weight_slices_index[indicator] = i;
  }

  for (size_t i = 0; i < input_slices_accessor.size(0); ++i) {
    auto indicator = input_slices_accessor[i][0];
    auto weight_index = weight_slices_index[indicator];
    auto input_start = input_slices_accessor[i][1];
    auto input_end = input_slices_accessor[i][2];
    auto weight_start = weight_slices_accessor[weight_index][1];
    auto weight_end = weight_slices_accessor[weight_index][2];
    auto sub_input_tensor = input.index({Slice(0, input_start, input_end)});
    auto sub_weight_tensor = weight.index({Slice(0, weight_start, weight_end)});
    auto sub_grad_tensor = grad.index({Slice(0, input_start, input_end)});
    auto sub_input_grad_tensor = input_grad.index({Slice(0, input_start, input_end)});
    auto sub_weight_grad_tensor = weight_grad.index({Slice(0, weight_start, weight_end)});
    sub_input_grad_tensor = torch::mm(sub_grad_tensor, sub_weight_tensor);
    sub_weight_grad_tensor = torch::mm(sub_grad_tensor, sub_input_tensor);
  }
}

}  // namespace fasten

#endif  // FASTEN_OPERATORS_DETAILS_BMM_TORCH