#include "ops.h"

#include <torch/extension.h>

#include "details/bmm_magma.h"
#include "details/bmm_native.h"
#include "details/bmm_torch.h"

namespace fasten {

torch::Tensor bmm_forward_handle(torch::Tensor input, torch::Tensor input_slices,
                                 torch::Tensor weight, torch::Tensor weight_slices,
                                 torch::Tensor output, Engine engine = Engine::TORCH) {
  auto input_slice_accessor = input_slices.accessor<long, 2>();
  auto weight_slice_accessor = weight_slices.accessor<long, 2>();
  auto weight_slice_index = fasten_slice_index_build(weight_slice_accessor);

  if (engine == Engine::TORCH) {
    bmm_forward<Engine::TORCH>(input, input_slice_accessor, weight, weight_slice_accessor,
                               weight_slice_index, output);
  }

  return output;
}

std::tuple<torch::Tensor, torch::Tensor> bmm_backward_handle(
    torch::Tensor grad, torch::Tensor input, torch::Tensor input_slices, torch::Tensor weight,
    torch::Tensor weight_slices, Engine engine = Engine::TORCH) {
  auto input_slice_accessor = input_slices.accessor<long, 2>();
  auto weight_slice_accessor = weight_slices.accessor<long, 2>();
  auto weight_slice_index = fasten_slice_index_build(weight_slice_accessor);
  auto input_grad = torch::zeros_like(input, input.options());
  auto weight_grad = torch::zeros_like(weight, weight.options());

  if (engine == Engine::TORCH) {
    bmm_backward<Engine::TORCH>(grad, input, input_slice_accessor, weight, weight_slice_accessor,
                                weight_slice_index, input_grad, weight_grad);
  }

  return {input_grad, weight_grad};
}

torch::Tensor bmul_forward_handle(torch::Tensor input, torch::Tensor input_slices,
                                  torch::Tensor weight, torch::Tensor weight_slices,
                                  torch::Tensor output) {
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> bmul_backward_handle(torch::Tensor grad,
                                                              torch::Tensor input,
                                                              torch::Tensor input_slices,
                                                              torch::Tensor weight,
                                                              torch::Tensor weight_slices) {
  return {torch::tensor({1}, grad.options()), torch::tensor({1}, grad.options())};
}

torch::Tensor bdiv_forward_handle(torch::Tensor input, torch::Tensor input_slices,
                                  torch::Tensor weight, torch::Tensor weight_slices,
                                  torch::Tensor output) {
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> bdiv_backward_handle(torch::Tensor grad,
                                                              torch::Tensor input,
                                                              torch::Tensor input_slices,
                                                              torch::Tensor weight,
                                                              torch::Tensor weight_slices) {
  return {torch::tensor({1}, grad.options()), torch::tensor({1}, grad.options())};
}

torch::Tensor bsub_forward_handle(torch::Tensor input, torch::Tensor input_slices,
                                  torch::Tensor weight, torch::Tensor weight_slices,
                                  torch::Tensor output) {
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> bsub_backward_handle(torch::Tensor grad,
                                                              torch::Tensor input,
                                                              torch::Tensor input_slices,
                                                              torch::Tensor weight,
                                                              torch::Tensor weight_slices) {
  return {torch::tensor({1}, grad.options()), torch::tensor({1}, grad.options())};
}

torch::Tensor badd_forward_handle(torch::Tensor input, torch::Tensor input_slices,
                                  torch::Tensor weight, torch::Tensor weight_slices,
                                  torch::Tensor output) {
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> badd_backward_handle(torch::Tensor grad,
                                                              torch::Tensor input,
                                                              torch::Tensor input_slices,
                                                              torch::Tensor weight,
                                                              torch::Tensor weight_slices) {
  return {torch::tensor({1}, grad.options()), torch::tensor({1}, grad.options())};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bmm_forward", &bmm_forward_handle, "Fasten bmm forward");
  m.def("bmm_backward", &bmm_backward_handle, "Fasten bmm backward");
  m.def("bmul_forward", &bmul_forward_handle, "Fasten mul forward");
  m.def("bmul_backward", &bmul_backward_handle, "Fasten mul backward");
  m.def("bdiv_forward", &bdiv_forward_handle, "Fasten div forward");
  m.def("bdiv_backward", &bdiv_backward_handle, "Fasten div backward");
  m.def("badd_forward", &badd_forward_handle, "Fasten add forward");
  m.def("badd_backward", &badd_backward_handle, "Fasten add backward");
  m.def("bsub_forward", &bsub_forward_handle, "Fasten sub forward");
  m.def("bsub_backward", &bsub_backward_handle, "Fasten sub backward");
  m.attr("MAX_TENSOR_DIMS") = MAX_TENSOR_DIMS;
  m.attr("MIN_TENSOR_DIMS") = MIN_TENSOR_DIMS;
  py::enum_<Engine>(m, "Engine", py::module_local())
      .value("TORCH", Engine::TORCH)
      .value("NATIVE", Engine::NATIVE)
      .value("MAGMA", Engine::MAGMA)
      .export_values();
}

}  // namespace fasten