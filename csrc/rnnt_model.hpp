#pragma once

#include <ATen/core/ivalue.h>
#include <string>
#include <torch/script.h>
#include <ATen/Parallel.h>
#include "rnnt_qsl.hpp"

using namespace torch::indexing;

namespace models {
//
// Functionality:
//   1. Model load&adjust
//
class TorchModel {
  using Module = torch::jit::script::Module;
  enum {
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

  template <typename... Args>
  std::vector<std::vector<int64_t>> inference(Args&&... args) {
    return greedy_decode(model_, std::forward<Args>(args)...);
  }

  template <typename... Args>
  std::vector<std::vector<int64_t>> inference_at(int socket, Args&&... args) {
    return greedy_decode(socket_model_[socket], std::forward<Args>(args)...);
  }

  template <typename... Args>
  std::vector<std::vector<int64_t>> greedy_decode(Module model, Args&&... args) {
    auto sub_modules = model.children();
    auto transcription = *sub_modules.begin();
    auto prediction = *(++sub_modules.begin());
    auto joint = *(++++sub_modules.begin());

    auto trans_res = transcription.forward(std::forward<Args>(args)...).toTuple()->elements();
    torch::Tensor f = trans_res[0].toTensor();
    torch::Tensor f_lens = trans_res[1].toTensor();

    auto batch_size = f_lens.size(0);
    std::vector<std::vector<int64_t>> res(batch_size);
    torch::Tensor eos_idxs = (f_lens - 1).to(torch::kLong);
    torch::Tensor time_idxs = torch::zeros(batch_size, torch::dtype(torch::kLong));

    torch::Tensor fi = f[0];
    torch::Tensor pre_g = torch::zeros({1, batch_size}, torch::dtype(torch::kLong));
    torch::Tensor pre_hg = torch::zeros({pred_num_layers, batch_size, pred_hidden_size});
    torch::Tensor pre_cg = torch::zeros({pred_num_layers, batch_size, pred_hidden_size});

    int loop = 0;
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

      // update decoder queue
      if (torch::count_nonzero(no_blank_idxs).item().toInt() != 0) {
      	pre_g.index_put_({0, no_blank_idxs}, symbols.index({no_blank_idxs}));
      	pre_hg.index_put_({0, no_blank_idxs, "..."}, hg.index({0, no_blank_idxs, "..."}));
      	pre_hg.index_put_({1, no_blank_idxs, "..."}, hg.index({1, no_blank_idxs, "..."}));
      	pre_cg.index_put_({0, no_blank_idxs, "..."}, cg.index({0, no_blank_idxs, "..."}));
      	pre_cg.index_put_({1, no_blank_idxs, "..."}, cg.index({1, no_blank_idxs, "..."}));
      }

      // update encoder queue
      if (torch::count_nonzero(no_blank_idxs).item().toInt() != batch_size) {
      	time_idxs += ~no_blank_idxs;
      	time_idxs = time_idxs.min(eos_idxs);  // TODO: add early response
      	if (torch::equal(time_idxs, eos_idxs))
      	  break;
      	auto fetch_idxs = time_idxs.unsqueeze(1).unsqueeze(0).expand({1, batch_size, trans_hidden_size});
      	fi = f.gather(0, fetch_idxs).squeeze(0);
      }
      loop += 1;
    }
    return res;
  }

private:
  torch::jit::script::Module model_;
  torch::jit::script::Module socket_model_[2];
};

}
