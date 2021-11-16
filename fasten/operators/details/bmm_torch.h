#ifndef FASTEN_OPERATORS_DETAILS_BMM_TORCH
#define FASTEN_OPERATORS_DETAILS_BMM_TORCH

#include <torch/extension.h>

#include <iostream>

#include "../ops.h"
#include "../utils.h"
namespace fasten {

using torch::indexing::Slice;

template <>
void bmm_forward<Engine::TORCH>(torch::Tensor input,
                                torch::TensorAccessor<long, 2> input_slice_accessor,
                                torch::Tensor weight,
                                torch::TensorAccessor<long, 2> weight_slice_accessor,
                                SliceIndex &weight_slice_index, torch::Tensor output) {
  for (auto i = 0; i < input_slice_accessor.size(0); ++i) {
    auto indicator = input_slice_accessor[i][0];
    auto weight_index = weight_slice_index[indicator];
    auto input_start = input_slice_accessor[i][1];
    auto input_end = input_slice_accessor[i][2];
    auto weight_start = weight_slice_accessor[weight_index][1];
    auto weight_end = weight_slice_accessor[weight_index][2];
    auto output_start = input_slice_accessor[i][1];
    auto output_end = input_slice_accessor[i][2];
    auto sub_input_tensor = input.index({Slice(input_start, input_end), Slice()});
    auto sub_weight_tensor = weight.index({Slice(weight_start, weight_end), Slice()});
    TORCH_CHECK(sub_weight_tensor.dim() == 2 || sub_weight_tensor.dim() == 3);
    if (sub_weight_tensor.dim() == 3) {
      sub_weight_tensor = torch::squeeze(sub_weight_tensor, 0);
    }
    auto sub_output_tensor = output.index({Slice(output_start, output_end), Slice()});
    torch::mm_out(sub_output_tensor, sub_input_tensor, sub_weight_tensor);
  }
}

template <>
void bmm_backward<Engine::TORCH>(torch::Tensor grad, torch::Tensor input,
                                 torch::TensorAccessor<long, 2> input_slice_accessor,
                                 torch::Tensor weight,
                                 torch::TensorAccessor<long, 2> weight_slice_accessor,
                                 SliceIndex &weight_slice_index, torch::Tensor input_grad,
                                 torch::Tensor weight_grad) {
  for (auto i = 0; i < input_slice_accessor.size(0); ++i) {
    auto indicator = input_slice_accessor[i][0];
    auto weight_index = weight_slice_index[indicator];
    auto input_start = input_slice_accessor[i][1];
    auto input_end = input_slice_accessor[i][2];
    auto weight_start = weight_slice_accessor[weight_index][1];
    auto weight_end = weight_slice_accessor[weight_index][2];
    auto sub_input_tensor = input.index({Slice(input_start, input_end), Slice()});
    auto sub_weight_tensor = weight.index({Slice(weight_start, weight_end), Slice()});
    auto sub_grad_tensor = grad.index({Slice(input_start, input_end), Slice()});
    auto sub_input_grad_tensor = weight_grad.index({Slice(input_start, input_end), Slice()});
    auto sub_weight_grad_tensor = weight_grad.index({Slice(weight_start, weight_end), Slice()});
    torch::mm_out(sub_input_grad_tensor, sub_grad_tensor, sub_weight_tensor);
    torch::addmm_out(sub_weight_grad_tensor, sub_weight_grad_tensor, sub_grad_tensor,
                     sub_input_tensor);
  }
}

}  // namespace fasten

#endif  // FASTEN_OPERATORS_DETAILS_BMM_TORCH