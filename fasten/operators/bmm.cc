#include "bmm.h"

#include <torch/extension.h>

#include <map>

#include "details/bmm_magma.h"
#include "details/bmm_native.h"
#include "details/bmm_torch.h"

at::Tensor fasten_bmm_forward(torch::Tensor input, torch::Tensor input_slices, torch::Tensor weight,
                              torch::Tensor weight_slices, BmmEngine engine = BmmEngine::TORCH) {
  TORCH_CHECK(engine == BmmEngine::TORCH, "fasten bmm on CPU only supports the TORCH engine");

  auto input_slices_accessor = input_slices.accessor<size_t, 2>();
  auto weight_slices_accessor = weight_slices.accessor<size_t, 2>();
  auto output = torch::empty({input.size(0), weight.size(-1)});

  bmm_forward_torch(input, input_slices_accessor, weight, weight_slices_accessor, output);
  return output;
}

std::vector<at::Tensor> fasten_bmm_backward(torch::Tensor grad, torch::Tensor input,
                                            torch::Tensor input_slices, torch::Tensor weight,
                                            torch::Tensor weight_slices,
                                            BmmEngine engine = BmmEngine::TORCH) {
  using torch::indexing::Slice;

  TORCH_CHECK(engine == BmmEngine::TORCH, "fasten bmm on CPU only supports the TORCH engine");

  auto input_slices_accessor = input_slices.accessor<size_t, 2>();
  auto weight_slices_accessor = weight_slices.accessor<size_t, 2>();
  auto input_grad = torch::empty_like(input);
  auto weight_grad = torch::empty_like(weight);

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

  return {input_grad, weight_grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fasten_bmm_forward, "LLTM forward");
  m.def("backward", &fasten_bmm_backward, "LLTM backward");
  py::enum_<BmmEngine>(m, "BmmEngine", py::module_local())
      .value("FASTEN_BMM_TORCH", BmmEngine::TORCH)
      .value("FASTEN_BMM_NATIVE", BmmEngine::NATIVE)
      .value("FASTEN_BMM_MAGMA", BmmEngine::MAGMA);
}