#pragma once

#include <ATen/core/ivalue.h>
#include <string>
#include <torch/script.h>
#include <ATen/Parallel.h>
#include "rnnt_transcription.hpp"

using namespace torch::indexing;

namespace models {
using Stack = std::vector<at::IValue>;
using Module = torch::jit::script::Module;

struct Transcription {
  Module pre_rnn;
  Module stack_time;
  Module post_rnn;
  Module pre_quantizer;
  Module post_quantizer;

  Transcription() {}

  Transcription(Module transcription) {
    auto module_ptr = transcription.children().begin();
    pre_rnn = *module_ptr++;
    stack_time = *module_ptr++;
    post_rnn = *module_ptr++;
    pre_quantizer = *module_ptr++;
    // post_quantizer = *module_ptr++;
  }

  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> forward (
      at::Tensor f, at::Tensor f_lens,
      at::Tensor pre_hx, at::Tensor pre_cx,
      at::Tensor post_hx, at::Tensor post_cx) {
    // 1. pre_rnn
    f = pre_quantizer({f}).toTensor();
    std::tie(f, pre_hx, pre_cx) = lstm_forward(pre_rnn, f, pre_hx, pre_cx, false);
    // 2. stack_time
    auto y = stack_time({f, f_lens}).toTensorList();
    f = y[0];
    f_lens = y[1];
    // 3. post_rnn
    // f = post_quantizer({f}).toTensor();
    std::tie(f, post_hx, post_cx) = lstm_forward(post_rnn, f, post_hx, post_cx, true);
    return {f, f_lens, pre_hx, pre_cx, post_hx, post_cx};
  }

  std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_forward (
      Module lstm, at::Tensor x, at::Tensor hx, at::Tensor cx, bool quant_last_layer) {
    auto cell_ptr = lstm.children().begin();
    at::Tensor hy_layer, cy_layer;
    std::vector<at::Tensor> hy_list, cy_list;
    auto num_layers = hx.sizes()[0];
    for (int64_t layer = 0; layer < num_layers; layer++) {
      std::tie(x, hy_layer, cy_layer) = lstm_layer_forward(
          *cell_ptr++, x, hx[layer], cx[layer], layer==(num_layers-1) && quant_last_layer);
      hy_list.emplace_back(hy_layer);
      cy_list.emplace_back(cy_layer);
    }
    auto hy = at::stack(hy_list, 0);
    auto cy = at::stack(cy_list, 0);
    return {x, hy, cy};
  }

  std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_layer_forward (
      Module lstm_cell, at::Tensor x, at::Tensor hx, at::Tensor cx, bool quant_y) {
    std::vector<at::Tensor> y_list;
    for (int64_t step = 0; step < x.sizes()[0]; step++) {
      auto outputs = lstm_cell({x[step], hx, cx, quant_y}).toTensorList();
      y_list.emplace_back(outputs[0]);
      hx = outputs[1];
      cx = outputs[2];
    }
    auto y = at::stack(y_list, 0);
    return {y, hx, cx};
  }
};

}
