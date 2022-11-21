#pragma once

#include <ATen/core/ivalue.h>
#include <string>
#include <torch/script.h>
#include <ATen/Parallel.h>
#include "rnnt_qsl.hpp"
#include "metadata.hpp"

namespace models {
using Module = torch::jit::script::Module;
using namespace torch::indexing;
using namespace rnnt;

struct RNNT {
  Module transcription;
  Module prediction;
  Module joint;
  Module update;

  RNNT() {}

  RNNT(Module model) {
    model.eval();
    auto module_ptr = model.children().begin();
    transcription = *module_ptr++;
    prediction = *module_ptr++;
    joint = *module_ptr++;
    update = *module_ptr;
  }
};

//
// Functionality:
//   1. Model load&adjust
//
class TorchModel {
public:
  TorchModel (const std::string filename, bool enable_bf16) {
    Module model = load(filename);
    model.eval();
    model_ = RNNT(model);
    socket_model_[0] = model_;
    socket_model_[1] = RNNT(model.clone());
    enable_bf16_ = enable_bf16;
  }

  TorchModel ();

  Module load (const std::string filename) {
    Module model = torch::jit::load(filename);
    return model;
  }

  void inference (State& state, int32_t split_len) {
    return greedy_decode(model_, state, state.finish_idx_, split_len);
  }

  void inference_at (int socket, State& state, int32_t split_len) {
    return greedy_decode(socket_model_[socket], state, state.finish_idx_, split_len);
  }

  void transcription_forward(RNNT model, State& state) {
    auto trans_res = model.transcription(
        {state.f_, state.f_lens_, state.pre_hx_ , state.pre_cx_ , state.post_hx_ , state.post_cx_}).toTuple()->elements();
    state.f_ = trans_res[0].toTensor();
    state.f_lens_ = trans_res[1].toTensor();
    state.pre_hx_  = trans_res[2].toTensorVector();
    state.pre_cx_  = trans_res[3].toTensorVector();
    state.post_hx_  = trans_res[4].toTensorVector();
    state.post_cx_  = trans_res[5].toTensorVector();
    if (enable_bf16_)
      state.f_ = state.f_.to(at::ScalarType::BFloat16);
  }

  void greedy_decode (RNNT model, State& state, at::Tensor finish_idx, int32_t split_len) {
    // init flags
    auto symbols_added = torch::zeros(state.batch_size_, torch::dtype(torch::kInt32));
    auto time_idx = torch::zeros(state.batch_size_, torch::dtype(torch::kInt32));
    // 1. do transcription
    if (split_len != -1) {
      // accumulate transcription
      TensorVector fi_list;
      fi_list.reserve(HALF_MAX_LEN);
      while(state.next()) {
        transcription_forward(model, state);
        fi_list.emplace_back(state.f_);
      }
      state.f_ = torch::cat(fi_list, 0);
      state.f_lens_ = torch::ceil(state.F_lens_ / 2).to(torch::kInt32);
    } else {
      transcription_forward(model, state);
    }
    auto fi = state.f_[0];

    while (true) {
      // 2. do prediction
      auto pred_res = model.prediction({state.pred_g_, std::make_tuple(state.pred_hg_, state.pred_cg_)}).toTuple()->elements();
      auto g = pred_res[0].toTensor();
      auto hg = pred_res[1].toTuple()->elements()[0].toTensor();
      auto cg = pred_res[1].toTuple()->elements()[1].toTensor();
      // 3. do joint
      auto y = model.joint({fi, g[0]}).toTensor();
      auto symbols = torch::argmax(y, 1);
      // 4. update state & flags
      bool finish = model.update({symbols, symbols_added, state.res_, state.res_idx_, time_idx,
          state.f_lens_, state.pred_g_, state.f_, fi, state.pred_hg_, state.pred_cg_, hg, cg}).toBool();
      if (finish) break;
    }
  }

  static std::string sequence_to_string(const at::Tensor& seq, const int32_t& seq_lens) {
    std::string str = "";
    for(int i = 0; i < seq_lens; i++)
      str.push_back(LABELS[seq[i].item().toInt()]);
    return str;
  }

private:
  RNNT model_;
  RNNT socket_model_[2];
  bool enable_bf16_ = true;
};

}
