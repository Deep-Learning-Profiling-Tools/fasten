#include "bmm.h"

#include <torch/extension.h>

#include "details/bmm_magma.h"
#include "details/bmm_native.h"
#include "details/bmm_torch.h"

namespace fasten {

torch::Tensor bmm_forward_handle(torch::Tensor input, torch::Tensor input_slices,
                                 torch::Tensor weight, torch::Tensor weight_slices,
                                 Engine engine = Engine::TORCH) {
  auto input_slice_accessor = input_slices.accessor<size_t, 2>();
  auto weight_slice_accessor = weight_slices.accessor<size_t, 2>();
  auto output = torch::zeros({input.size(0), weight.size(-1)}, input.options());

  if (engine == Engine::TORCH) {
    bmm_forward<Engine::TORCH>(input, input_slice_accessor, weight, weight_slice_accessor, output);
  }

  return output;
}

std::tuple<torch::Tensor, torch::Tensor> bmm_backward_handle(
    torch::Tensor grad, torch::Tensor input, torch::Tensor input_slices, torch::Tensor weight,
    torch::Tensor weight_slices, Engine engine = Engine::TORCH) {
  auto input_slice_accessor = input_slices.accessor<size_t, 2>();
  auto weight_slice_accessor = weight_slices.accessor<size_t, 2>();
  auto input_grad = torch::zeros_like(input, input.options());
  auto weight_grad = torch::zeros_like(weight, weight.options());

  if (engine == Engine::TORCH) {
    bmm_backward<Engine::TORCH>(grad, input, input_slice_accessor, weight, weight_slice_accessor,
                                input_grad, weight_grad);
  }

  return {input_grad, weight_grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &bmm_forward_handle, "Fasten BMM forward");
  m.def("backward", &bmm_backward_handle, "Fasten BMM backward");
  py::enum_<Engine>(m, "Engine", py::module_local())
      .value("TORCH", Engine::TORCH)
      .value("NATIVE", Engine::NATIVE)
      .value("MAGMA", Engine::MAGMA)
      .export_values();
}

}  // namespace fasten