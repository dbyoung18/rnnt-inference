#pragma once

#include <ATen/core/ivalue.h>
#include <string>
#include <torch/script.h>
#include <ATen/Parallel.h>
#include "rnnt_qsl.hpp"

using namespace torch::indexing;

namespace models {
using Module = torch::jit::script::Module;

static const std::vector<char> LABELS = {
  ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\''};

struct RNNT {
  Module transcription;
  Module prediction;
  Module joint;

  RNNT() {}

  RNNT(Module model) {
    model.eval();
    auto module_ptr = model.children().begin();
    transcription = *module_ptr++;
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
  TorchModel (const std::string filename, int64_t split_len, bool enable_bf16) {
    Module model = load(filename);
    model.eval();
    model_ = RNNT(model);
    socket_model_[0] = model_;
    socket_model_[1] = RNNT(model.clone());
    split_len_ = split_len;
    enable_bf16_ = enable_bf16;
  }

  TorchModel ();

  Module load (const std::string filename) {
    Module model = torch::jit::load(filename);
    return model;
  }

  std::vector<at::Tensor> inference (qsl::Stack inputs) {
    return forward(model_, inputs);
  }

  std::vector<at::Tensor> inference_at (int socket, qsl::Stack inputs) {
    return forward(socket_model_[socket], inputs);
  }

  std::vector<at::Tensor> forward (RNNT model, qsl::Stack inputs) {
    auto x = inputs[0].toTensor();
    auto x_lens = inputs[1].toTensor();
    auto batch_size = x_lens.size(0);
    auto res = torch::full({batch_size, x_lens.max().item().toInt()*3}, SOS, torch::dtype(torch::kInt64));
    auto res_idx = torch::full({batch_size}, -1, torch::dtype(torch::kInt64));
    // init transcription tensors
    std::vector<at::Tensor> pre_hx(pre_num_layers);
    std::vector<at::Tensor> pre_cx(pre_num_layers);
    std::vector<at::Tensor> post_hx(post_num_layers);
    std::vector<at::Tensor> post_cx(post_num_layers);
    for(int i = 0; i < pre_num_layers; i++){
      pre_hx[i] = torch::zeros(
          {batch_size, trans_hidden_size}, torch::dtype(torch::kInt8));
      pre_cx[i] = torch::zeros(
          {batch_size, trans_hidden_size}, torch::dtype(torch::kFloat16));
    }
    for(int i = 0; i < post_num_layers; i++){
      post_hx[i] = torch::zeros(
          {batch_size, trans_hidden_size}, torch::dtype(torch::kInt8));
      post_cx[i] = torch::zeros(
          {batch_size, trans_hidden_size}, torch::dtype(torch::kFloat16));
    }

    // init prediction tensors
    auto pred_g = torch::full({1, batch_size}, SOS, torch::dtype(torch::kLong));
    auto pred_dtype_ = enable_bf16_ ? at::ScalarType::BFloat16 : torch::kFloat32;
    auto pred_hg = torch::zeros({pred_num_layers, batch_size, pred_hidden_size}, torch::dtype(pred_dtype_));
    auto pred_cg = torch::zeros({pred_num_layers, batch_size, pred_hidden_size}, torch::dtype(pred_dtype_));

    if (split_len_ != -1) {
      auto max_len = x_lens.max().item().toInt();
      auto split_lens = at::full({batch_size}, split_len_, torch::dtype(torch::kLong));
      for (int64_t split_idx = 0; split_idx < max_len; split_idx += split_len_) {
        // 0. split x, x_lens
        auto xi_lens = torch::min(split_lens, (x_lens - split_idx).clamp(0));
        auto xi = x.index({Slice(split_idx, split_idx + split_len_, None)});
        greedy_decode(model, xi, xi_lens, pre_hx, pre_cx, post_hx, post_cx, pred_g, pred_hg, pred_cg, res, res_idx);
      }
    } else {
      greedy_decode(model, x, x_lens, pre_hx, pre_cx, post_hx, post_cx, pred_g, pred_hg, pred_cg, res, res_idx);
    }
    return {res, res_idx+1};
  }

  void greedy_decode (
      RNNT model, at::Tensor f, at::Tensor f_lens,
      std::vector<at::Tensor>& pre_hx, std::vector<at::Tensor>& pre_cx,
      std::vector<at::Tensor>& post_hx, std::vector<at::Tensor>& post_cx,
      at::Tensor& pred_g, at::Tensor& pred_hg, at::Tensor& pred_cg,
      at::Tensor& res,
      at::Tensor& res_idx) {
    // init flags
    auto batch_size = f_lens.size(0);
    auto symbols_added = torch::zeros(batch_size, torch::dtype(torch::kLong));
    auto time_idx = torch::zeros(batch_size, torch::dtype(torch::kLong));
    auto finish = f_lens.eq(0);
    auto fi_idx = torch::range(0, batch_size-1, torch::dtype(torch::kLong));
    // 1. do transcription
    auto trans_res = model.transcription({f, f_lens, std::make_tuple(pre_hx, pre_cx), std::make_tuple(post_hx, post_cx)}).toTuple()->elements();
    f = trans_res[0].toTensor();
    f_lens = trans_res[1].toTensor();
    pre_hx = trans_res[2].toTuple()->elements()[0].toTensorList().vec();
    pre_cx = trans_res[2].toTuple()->elements()[1].toTensorList().vec();
    post_hx = trans_res[3].toTuple()->elements()[0].toTensorList().vec();
    post_cx = trans_res[3].toTuple()->elements()[1].toTensorList().vec();
    auto eos_idx = (f_lens-1).clamp(0);
    if (enable_bf16_)
      f = f.to(at::ScalarType::BFloat16);
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
      if (torch::any(update_g).item().toBool()) {
        res_idx += update_g;
        // 4.1. update res
        res.index_put_({update_g, res_idx.index({update_g})}, symbols.index({update_g}));
        // 4.2. update symbols_added
        symbols_added += update_g;
        // 4.3. update g
        pred_g.index_put_({0, update_g}, symbols.index({update_g}));
        pred_hg.index_put_({Slice(0), update_g, "..."}, hg.index({Slice(0), update_g, "..."}));
        pred_cg.index_put_({Slice(0), update_g, "..."}, cg.index({Slice(0), update_g, "..."}));
      }
      // 5. if (BLANK or MAX_SYMBOLS_PER_STEP) and no FINISH
      auto update_f = ~update_g & ~finish;
      if (torch::any(update_f).item().toBool()) {
        // 5.1. update time_idx
        time_idx += update_f;
        // 5.2. BCE
        finish |= time_idx.ge(f_lens);
        time_idx = time_idx.min(eos_idx);
        if (torch::all(finish).item().toBool())
          break;
        // 5.3. update f
        fi = f.index({time_idx, fi_idx, "..."});
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

  static std::string sequence_to_string(const at::Tensor& seq, const int64_t& seq_lens) {
    std::string str = "";
    for(int i=0;i<seq_lens;i++)
      str.push_back(LABELS[seq[i].item().toInt()]);
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
  bool enable_bf16_ = true;
};

}
