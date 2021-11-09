#include "bmm.h"

#include <torch/extension.h>

#include <map>

#include "details/bmm_magma.h"
#include "details/bmm_native.h"
#include "details/bmm_torch.h"

namespace fasten {

torch::Tensor fasten_bmm_forward(torch::Tensor input, torch::Tensor input_slices,
                                 torch::Tensor weight, torch::Tensor weight_slices,
                                 BmmEngine engine = BmmEngine::TORCH) {
  TORCH_CHECK(engine == BmmEngine::TORCH, "fasten bmm on CPU only supports the TORCH engine");

  auto input_slice_accessor = input_slices.accessor<size_t, 2>();
  auto weight_slice_accessor = weight_slices.accessor<size_t, 2>();
  auto output = torch::zeros({input.size(0), weight.size(-1)}, input.options());

  bmm_forward<engine>(input, input_slice_accessor, weight, weight_slice_accessor, output);
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> fasten_bmm_backward(
    torch::Tensor grad, torch::Tensor input, torch::Tensor input_slices, torch::Tensor weight,
    torch::Tensor weight_slices, torch::Tensor input_grad, torch::Tensor weight_grad,
    BmmEngine engine = BmmEngine::TORCH) {
  TORCH_CHECK(engine == BmmEngine::TORCH, "fasten bmm on CPU only supports the TORCH engine");

  auto input_slice_accessor = input_slices.accessor<size_t, 2>();
  auto weight_slice_accessor = weight_slices.accessor<size_t, 2>();
  auto input_grad = torch::zeros_like(input, input.options());
  auto weight_grad = torch::zeros_like(weight, weight.options());

  bmm_backward<engine>(grad, input, input_slice_accessor, weight, weight_slice_accessor, input_grad,
                       weight_grad);

  return {input_grad, weight_grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fasten_bmm_forward, "Fasten BMM forward");
  m.def("backward", &fasten_bmm_backward, "Fasten BMM backward");
  py::enum_<BmmEngine>(m, "BmmEngine", py::module_local())
      .value("FASTEN_BMM_TORCH", BmmEngine::TORCH)
      .value("FASTEN_BMM_NATIVE", BmmEngine::NATIVE)
      .value("FASTEN_BMM_MAGMA", BmmEngine::MAGMA);
}

}  // namespace fasten