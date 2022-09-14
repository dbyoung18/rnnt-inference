#pragma once

#include <ATen/core/ivalue.h>
#include <string>
#include <torch/script.h>
#include <ATen/Parallel.h>
#include "rnnt_qsl.hpp"
#include "rnnt_transcription.hpp"

using namespace torch::indexing;

namespace models {
using Stack = std::vector<at::IValue>;
using Module = torch::jit::script::Module;

static const std::vector<char> LABELS = {
  ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\''};

struct RNNT {
  Transcription transcription;
  Module prediction;
  Module joint;

  RNNT() {}

  RNNT(Module model) {
    model.eval();
    auto module_ptr = model.children().begin();
    transcription = Transcription(*module_ptr++);
    prediction = *module_ptr++;
    joint = *module_ptr;
  }
};

//
// Functionality:
//   1. Model load&adjust
//
class TorchModel {
  using Module = torch::jit::script::Module;

public:
  TorchModel (const std::string filename, int64_t split_len) {
    Module model = load(filename);
    model.eval();
    model_ = RNNT(model);
    socket_model_[0] = model_;
    socket_model_[1] = RNNT(model.clone());
    split_len_ = split_len;
  }

  TorchModel ();

  Module load (const std::string filename) {
    Module model = torch::jit::load(filename);
    return model;
  }

  std::vector<std::vector<int64_t>> inference (Stack inputs) {
    return forward(model_, inputs);
  }

  std::vector<std::vector<int64_t>> inference_at (int socket, Stack inputs) {
    return forward(socket_model_[socket], inputs);
  }

  std::vector<std::vector<int64_t>> forward (RNNT model, Stack inputs) {
    auto x = inputs[0].toTensor();
    auto x_lens = inputs[1].toTensor();
    auto batch_size = x_lens.size(0);
    std::vector<std::vector<int64_t>> res(batch_size);
    // init transcription tensors
    auto pre_hx = torch::zeros(
        {pre_num_layers, batch_size, trans_hidden_size}, torch::dtype(torch::kInt8));
    auto pre_cx = torch::zeros(
        {pre_num_layers, batch_size, trans_hidden_size}, torch::dtype(torch::kInt8));
    auto post_hx = torch::zeros(
        {post_num_layers, batch_size, trans_hidden_size}, torch::dtype(torch::kInt8));
    auto post_cx = torch::zeros(
        {post_num_layers, batch_size, trans_hidden_size}, torch::dtype(torch::kInt8));
    // init prediction tensors
    auto pred_g = torch::full({1, batch_size}, SOS, torch::dtype(torch::kLong));
    auto pred_hg = torch::zeros({pred_num_layers, batch_size, pred_hidden_size});
    auto pred_cg = torch::zeros({pred_num_layers, batch_size, pred_hidden_size});

    if (split_len_ != -1) {
      auto max_len = x_lens.max().item().toInt();
      auto split_lens = at::full({batch_size}, split_len_, torch::dtype(torch::kLong));
      for (int64_t split_idx = 0; split_idx < max_len; split_idx += split_len_) {
        // 0. split x, x_lens
        auto xi_lens = torch::min(split_lens, (x_lens - split_idx).clamp(0));
        auto xi = x.index({Slice(split_idx, split_idx + split_len_, None)});
        greedy_decode(model, xi, xi_lens, pre_hx, pre_cx, post_hx, post_cx, pred_g, pred_hg, pred_cg, res);
      }
    } else {
      greedy_decode(model, x, x_lens, pre_hx, pre_cx, post_hx, post_cx, pred_g, pred_hg, pred_cg, res);
    }
    return res;
  }

  void greedy_decode (
      RNNT model, at::Tensor f, at::Tensor f_lens,
      at::Tensor& pre_hx, at::Tensor& pre_cx,
      at::Tensor& post_hx, at::Tensor& post_cx,
      at::Tensor& pred_g, at::Tensor& pred_hg, at::Tensor& pred_cg,
      std::vector<std::vector<int64_t>>& res) {
    // init flags
    auto batch_size = f_lens.size(0);
    auto symbols_added = torch::zeros(batch_size, torch::dtype(torch::kLong));
    auto time_idx = torch::zeros(batch_size, torch::dtype(torch::kLong));
    auto finish = f_lens.eq(0);
    // 1. do transcription
    std::tie(f, f_lens, pre_hx, pre_cx, post_hx, post_cx) = model.transcription.forward(f, f_lens, pre_hx, pre_cx, post_hx, post_cx);
    auto fi = f[0];

    while (true) {
      // 2. do prediction
      auto pred_res = model.prediction({pred_g, std::make_tuple(pred_hg, pred_cg)}).toTuple()->elements();
      auto g = pred_res[0].toTensor();
      auto hg = pred_res[1].toTuple()->elements()[0].toTensor();
      auto cg = pred_res[1].toTuple()->elements()[1].toTensor();
      // 3. do joint
      auto y = model.joint({fi, g[0]}).toTensor();
      auto symbols = torch::argmax(y, 1);
      // 4. if (no BLANK and no MAX_SYMBOLS_PER_STEP) and no FINISH
      auto update_g = symbols.ne(BLANK) & symbols_added.ne(max_symbols_per_step) & ~finish;
      if (torch::count_nonzero(update_g).item().toInt() != 0) {
        // 4.1. update res
        for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx)
          if (update_g[batch_idx].item().toBool() == true)
            res[batch_idx].push_back(symbols[batch_idx].item().toInt());
        // 4.2. update symbols_added
        symbols_added += update_g;
        // 4.3. update g
        pred_g.index_put_({0, update_g}, symbols.index({update_g}));
        pred_hg.index_put_({0, update_g, "..."}, hg.index({0, update_g, "..."}));
        pred_hg.index_put_({1, update_g, "..."}, hg.index({1, update_g, "..."}));
        pred_cg.index_put_({0, update_g, "..."}, cg.index({0, update_g, "..."}));
        pred_cg.index_put_({1, update_g, "..."}, cg.index({1, update_g, "..."}));
      }
      // 5. if (BLANK or MAX_SYMBOLS_PER_STEP) and no FINISH
      auto update_f = ~update_g & ~finish;
      if (torch::count_nonzero(update_f).item().toInt() != 0) {
        // 5.1. update time_idx
        time_idx += update_f;
        // 5.2. BCE
        finish |= time_idx.ge(f_lens);
        time_idx = time_idx.min(f_lens-1).clamp(0);
        if (torch::count_nonzero(finish).item().toInt() == batch_size)
          break;
        // 5.3. update f
        auto fetch_idx = time_idx.unsqueeze(1).unsqueeze(0).expand(
            {1, batch_size, trans_hidden_size});
        fi = f.gather(0, fetch_idx).squeeze(0);
        // 5.4. reset symbols_added
        symbols_added *= ~update_f;
      }
    }
  }

  static std::string sequence_to_string(const std::vector<int64_t>& seq) {
    std::string str = "";
    for (auto ch : seq)
      str.push_back(LABELS[ch]);
    return str;
  }

private:
  enum {
    pre_num_layers = 2,
    post_num_layers = 3,
    trans_hidden_size = 1024,
    pred_num_layers = 2,
    pred_hidden_size = 320,
    joint_hidden_size = 512,
    SOS = -1,
    BLANK = 28,
    max_symbols_per_step = 30,
  };

  RNNT model_;
  RNNT socket_model_[2];
  int64_t split_len_ = -1;
};

}
