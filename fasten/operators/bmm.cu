#include "bmm.h"

#include <torch/extension.h>

at::Tensor fasten_bmm_forward_cuda(torch::Tensor input, torch::Tensor input_slices, torch::Tensor weight,
																	 torch::Tensor weight_slices, BmmEngine engine = BmmEngine::TORCH) {
  auto input_slices_accessor = input_slices.accessor<size_t, 2>();
  auto weight_slices_accessor = weight_slices.accessor<size_t, 2>();
  auto output = torch::empty({input.size(0), weight.size(-1)}, torch::TensorOptions().device(torch::kCUDA));

  std::map<size_t, size_t> weight_slices_index;
  for (size_t i = 0; i < weight_slices_accessor.size(0); ++i) {
    size_t indicator = weight_slices_accessor[i][0];
    weight_slices_index[indicator] = i;
  }

  bmm_forward_torch(input, input_slices_accessor, weight, weight_slices_accessor,
                    weight_slices_index, output);
  return output;
}

std::vector<at::Tensor> fasten_bmm_backward(torch::Tensor grad, torch::Tensor input,
                                            torch::Tensor input_slices, torch::Tensor weight,
                                            torch::Tensor weight_slices, BmmEngine engine = BmmEngine::TORCH) {
}