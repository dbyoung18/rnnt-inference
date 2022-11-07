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

  std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<at::Tensor>> forward (
      at::Tensor f, at::Tensor f_lens,
      std::vector<at::Tensor> pre_hx, std::vector<at::Tensor> pre_cx,
      std::vector<at::Tensor> post_hx, std::vector<at::Tensor> post_cx) {
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

  std::tuple<at::Tensor, std::vector<at::Tensor>, std::vector<at::Tensor>> lstm_forward (
      Module lstm, at::Tensor x, std::vector<at::Tensor> hx, std::vector<at::Tensor> cx, bool skip_quant_y) {
      auto layer_ptr = lstm.children().begin();
      auto num_layers = hx.size();
      for(int64_t layer = 0; layer < num_layers; layer++){
        auto skip_quant = (layer == (num_layers-1)) && skip_quant_y;
        auto output = (*layer_ptr++)({x, hx[layer], cx[layer], skip_quant}).toTuple()->elements();
        x = output[0].toTensor();
        hx[layer] = output[1].toTensor();
        cx[layer] = output[2].toTensor();
      }
      return {x, hx, cx};
    }
};

}
