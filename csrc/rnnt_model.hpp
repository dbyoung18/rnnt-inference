#pragma once

#include <ATen/core/ivalue.h>
#include <string>
#include <torch/script.h>
#include <ATen/Parallel.h>
#include "rnnt_qsl.hpp"

using namespace torch::indexing;

namespace models {
using Stack = std::vector<at::IValue>;
//
// Functionality:
//   1. Model load&adjust
//
class TorchModel {
  using Module = torch::jit::script::Module;
  enum {
    pre_num_layers = 2,
    post_num_layers = 3,
    trans_hidden_size = 1024,
    pred_num_layers = 2,
    pred_hidden_size = 320,
    joint_num_layers = 512,
    BLANK = 28
  };

public:
  TorchModel (const std::string filename) : model_(torch::jit::load(filename)) {
    model_.eval();
    socket_model_[0] = model_;
    socket_model_[1] = model_.clone();
  }

  TorchModel ();

  void load(const std::string filename) {
    model_ = torch::jit::load(filename);
  }

  std::vector<std::vector<int64_t>> inference(Stack inputs) {
    return greedy_decode(model_, inputs);
  }

  std::vector<std::vector<int64_t>> inference_at(int socket, Stack inputs) {
    return greedy_decode(socket_model_[socket], inputs);
  }

  std::vector<at::Tensor> lstm_layer_forward(Module lstm_cell, at::Tensor x,
      at::Tensor hx, at::Tensor cx, bool last_layer) {
    at::Tensor cell_y;
    std::vector<at::Tensor> y_list;
    for (int64_t step = 0; step < x.sizes()[0]; step++) {
      auto outputs = lstm_cell({x[step], hx, cx, last_layer}).toTensorList();
      cell_y = outputs[0];
      hx = outputs[1];
      cx = outputs[2];
      y_list.push_back(cell_y);
    }
    auto y = at::stack(y_list, 0);
    return {y, hx, cx};
  }

  std::vector<at::Tensor> lstm_forward(Module lstm, at::Tensor x,
      at::Tensor hx, at::Tensor cx, int64_t num_layers) {
    auto sub_modules = lstm.children();
    auto cell_ptr = sub_modules.begin();
    std::vector<at::Tensor> hy_list(num_layers);
    std::vector<at::Tensor> cy_list(num_layers);
    for (int64_t layer = 0; layer < num_layers; layer++) {
      //auto lstm_cell = *sub_modules.begin();
      auto lstm_cell = *cell_ptr;
      auto outputs = lstm_layer_forward(
          lstm_cell, x, hx[layer], cx[layer], layer==(num_layers-1));
      x = outputs[0];
      hy_list[layer] = outputs[1];
      cy_list[layer] = outputs[2];
      cell_ptr = cell_ptr++;
    }
    auto hy = at::stack(hy_list, 0);
    auto cy = at::stack(cy_list, 0);
    return {x, hy, cy};
  }

  std::vector<at::Tensor> trans_forward(Module transcription, Stack inputs) {
    auto sub_modules = transcription.children();
    auto pre_rnn = *sub_modules.begin();
    auto stack_time = *(++sub_modules.begin());
    auto post_rnn = *(++++sub_modules.begin());
    auto pre_quantizer = *(++++++sub_modules.begin());
    auto post_quantizer = *(++++++++sub_modules.begin());

    auto x = inputs[0];
    auto x_lens = inputs[1];

    auto quant_x = pre_quantizer({x}).toTensor();
    auto hx = torch::zeros({pre_num_layers, quant_x.size(1), trans_hidden_size},
        torch::dtype(torch::kInt8));
    auto cx = torch::zeros({pre_num_layers, quant_x.size(1), trans_hidden_size},
        torch::dtype(torch::kFloat32));
    auto y1 = lstm_forward(pre_rnn, quant_x, hx, cx, pre_num_layers);

    auto y2 = stack_time({y1[0], x_lens}).toTensorList();

    auto quant_x2 = post_quantizer({y2.get(0)}).toTensor().contiguous();
    auto hx2 = torch::zeros(
        {post_num_layers, quant_x2.size(1), trans_hidden_size},
        torch::dtype(torch::kInt8));
    auto cx2 = torch::zeros(
        {post_num_layers, quant_x2.size(1), trans_hidden_size},
        torch::dtype(torch::kFloat32));
    auto f = lstm_forward(post_rnn, quant_x2, hx2, cx2, post_num_layers); 
    return {f[0], y2[1]};
  }

  std::vector<std::vector<int64_t>> greedy_decode(Module model, Stack inputs) {
    auto sub_modules = model.children();
    auto transcription = *sub_modules.begin();
    auto prediction = *(++sub_modules.begin());
    auto joint = *(++++sub_modules.begin());

    auto trans_res = trans_forward(transcription, inputs);
    //auto trans_res = transcription.forward(inputs).toTuple()->elements();
    torch::Tensor f = trans_res[0];
    torch::Tensor f_lens = trans_res[1];

    auto batch_size = f_lens.size(0);
    std::vector<std::vector<int64_t>> res(batch_size);
    torch::Tensor eos_idxs = (f_lens - 1).to(torch::kLong);
    torch::Tensor time_idxs = torch::zeros(batch_size, torch::dtype(torch::kLong));

    torch::Tensor fi = f[0];
    torch::Tensor pre_g = torch::zeros({1, batch_size}, torch::dtype(torch::kLong));
    torch::Tensor pre_hg = torch::zeros({pred_num_layers, batch_size, pred_hidden_size});
    torch::Tensor pre_cg = torch::zeros({pred_num_layers, batch_size, pred_hidden_size});

    while (true) {
      auto pred_res = prediction({pre_g, std::make_tuple(pre_hg, pre_cg)}).toTuple()->elements();
      auto g = pred_res[0].toTensor();
      auto hg = pred_res[1].toTuple()->elements()[0].toTensor();
      auto cg = pred_res[1].toTuple()->elements()[1].toTensor();

      auto y = joint({fi, g[0]}).toTensor();
      auto symbols = torch::argmax(y, 1);

      // update res
      auto no_blank_idxs = torch::ne(symbols, BLANK);
      if (torch::count_nonzero(no_blank_idxs).item().toInt() != 0) {
      	for (int64_t i = 0; i < batch_size; ++i)
      	  if (no_blank_idxs[i].item().toBool() == true)
      	    res[i].push_back(symbols[i].item().toInt());
      }

      // update g
      if (torch::count_nonzero(no_blank_idxs).item().toInt() != 0) {
      	pre_g.index_put_({0, no_blank_idxs}, symbols.index({no_blank_idxs}));
      	pre_hg.index_put_({0, no_blank_idxs, "..."}, hg.index({0, no_blank_idxs, "..."}));
      	pre_hg.index_put_({1, no_blank_idxs, "..."}, hg.index({1, no_blank_idxs, "..."}));
      	pre_cg.index_put_({0, no_blank_idxs, "..."}, cg.index({0, no_blank_idxs, "..."}));
      	pre_cg.index_put_({1, no_blank_idxs, "..."}, cg.index({1, no_blank_idxs, "..."}));
      }

      // update f
      if (torch::count_nonzero(no_blank_idxs).item().toInt() != batch_size) {
      	time_idxs += ~no_blank_idxs;
      	time_idxs = time_idxs.min(eos_idxs);  // TODO: add early response
      	if (torch::equal(time_idxs, eos_idxs))
      	  break;
      	auto fetch_idxs = time_idxs.unsqueeze(1).unsqueeze(0).expand({1, batch_size, trans_hidden_size});
      	fi = f.gather(0, fetch_idxs).squeeze(0);
      }
    }
    return res;
  }

private:
  torch::jit::script::Module model_;
  torch::jit::script::Module socket_model_[2];
};

}
