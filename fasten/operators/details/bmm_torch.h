#include <torch/extension.h>

at::Tensor bmm_forward_torch(const torch::Tensor input,
                             const torch::TensorAccessor<size_t, 2> input_slices_accessor,
                             const torch::Tensor weight,
                             const torch::TensorAccessor<size_t, 2> weight_slices_accessor,
                             torch::Tensor output) {
  std::map<size_t, size_t> weight_slices_index;
  for (size_t i = 0; i < weight_slices_accessor.size(0); ++i) {
    size_t indicator = weight_slices_accessor[i][0];
    weight_slices_index[indicator] = i;
  }

  using torch::indexing::Slice;
  for (size_t i = 0; i < input_slices_accessor.size(0); ++i) {
    // Assume there's always a corresponding weight_index
    // auto iter = weight_slices_index.find(indicator);
    // if (iter == weight_slices_index.end()) {
    //  continue;
    //}
    // auto weight_index = iter->second;
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
