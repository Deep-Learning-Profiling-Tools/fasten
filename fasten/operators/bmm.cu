#include <torch/extension.h>

#include "bmm.h"
#include "details/bmm_torch.h"

namespace fasten {

at::Tensor fasten_bmm_forward_cuda(torch::Tensor input, torch::Tensor input_slices,
                                   torch::Tensor weight, torch::Tensor weight_slices,
                                   BmmEngine engine = BmmEngine::TORCH) {
  auto input_slices_accessor = input_slices.accessor<size_t, 2>();
  auto weight_slices_accessor = weight_slices.accessor<size_t, 2>();
  auto output = torch::zeros({input.size(0), weight.size(-1)}, input.options());

  bmm_forward<engine>(input, input_slices_accessor, weight, weight_slices_accessor, output);
  return output;
}

std::tuple<at::Tensor, at::Tensor> fasten_bmm_backward_cuda(torch::Tensor grad, torch::Tensor input,
                                                            torch::Tensor input_slices,
                                                            torch::Tensor weight,
                                                            torch::Tensor weight_slices,
                                                            BmmEngine engine = BmmEngine::TORCH) {
  auto input_slices_accessor = input_slices.accessor<size_t, 2>();
  auto weight_slices_accessor = weight_slices.accessor<size_t, 2>();
  auto input_grad = torch::zeros_like(input, input.options());
  auto weight_grad = torch::zeros_like(weight, weight.options());

  bmm_backward<engine>(grad, input, input_slices_accessor, weight, weight_slices_accessor,
                       input_grad, weight_grad);

  return {input_grad, weight_grad};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fasten_bmm_forward, "Fasten BMM forward (CUDA)");
  m.def("backward", &fasten_bmm_backward, "Fasten BMM backward (CUDA)");
  py::enum_<BmmEngine>(m, "BmmEngine", py::module_local())
      .value("FASTEN_BMM_TORCH", BmmEngine::TORCH)
      .value("FASTEN_BMM_NATIVE", BmmEngine::NATIVE)
      .value("FASTEN_BMM_MAGMA", BmmEngine::MAGMA);
}

}  // namespace fasten