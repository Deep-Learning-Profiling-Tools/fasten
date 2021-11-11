#ifndef FASTEN_OPERATORS_DETAILS_BMM_TORCH
#define FASTEN_OPERATORS_DETAILS_BMM_TORCH

#include <torch/extension.h>

#include "../bmm.h"
#include "../utils.h"

namespace fasten {

template <>
void bmm_forward<Engine::TORCH>(torch::Tensor input,
                                torch::TensorAccessor<long, 2> input_slice_accessor,
                                torch::Tensor weight,
                                torch::TensorAccessor<long, 2> weight_slice_accessor,
                                torch::Tensor output) {
  auto weight_slice_index = fasten_slice_index_build(weight_slice_accessor);

  using torch::indexing::Slice;
  for (auto i = 0; i < input_slice_accessor.size(0); ++i) {
    auto indicator = input_slice_accessor[i][0];
    auto weight_index = weight_slice_index[indicator];
    auto input_start = input_slice_accessor[i][1];
    auto input_end = input_slice_accessor[i][2];
    auto weight_start = weight_slice_accessor[weight_index][1];
    auto weight_end = weight_slice_accessor[weight_index][2];
    auto sub_input_tensor = input.index({Slice(0, input_start, input_end)});
    auto sub_weight_tensor = weight.index({Slice(0, weight_start, weight_end)});
    auto sub_output_tensor = output.index({Slice(0, input_start, input_end)});
    sub_output_tensor = torch::mm(sub_input_tensor, sub_weight_tensor);
  }
}

template <>
void bmm_backward<Engine::TORCH>(torch::Tensor grad, torch::Tensor input,
                                 torch::TensorAccessor<long, 2> input_slice_accessor,
                                 torch::Tensor weight,
                                 torch::TensorAccessor<long, 2> weight_slice_accessor,
                                 torch::Tensor input_grad, torch::Tensor weight_grad) {
  using torch::indexing::Slice;

  auto weight_slice_index = fasten_slice_index_build(weight_slice_accessor);

  for (auto i = 0; i < input_slice_accessor.size(0); ++i) {
    auto indicator = input_slice_accessor[i][0];
    auto weight_index = weight_slice_index[indicator];
    auto input_start = input_slice_accessor[i][1];
    auto input_end = input_slice_accessor[i][2];
    auto weight_start = weight_slice_accessor[weight_index][1];
    auto weight_end = weight_slice_accessor[weight_index][2];
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