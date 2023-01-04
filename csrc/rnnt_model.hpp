#pragma once

#include <ATen/core/ivalue.h>
#include <string>
#include <torch/script.h>
#include <ATen/Parallel.h>
#include "rnnt_qsl.hpp"
#include "metadata.hpp"

namespace rnnt {
namespace models {
using Module = torch::jit::script::Module;
using namespace torch::indexing;

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
  TorchModel (const std::string filename) {
    Module model = load(filename);
    model.eval();
    model_ = RNNT(model);
    socket_model_[0] = model_;
    socket_model_[1] = RNNT(model.clone());
  }

  TorchModel ();

  Module load (const std::string filename) {
    Module model = torch::jit::load(filename);
    return model;
  }

  void inference_at (int socket, State& state) {
    transcription_encode(socket, state);
    greedy_decode(socket, state);
    return;
  }

  void transcription_encode (int socket, State& state) {
    auto trans_res = socket_model_[socket].transcription(
        {state.f_, state.f_lens_, state.pre_hx_, state.pre_cx_, state.post_hx_, state.post_cx_}).toTuple()->elements();
    state.f_ = trans_res[0].toTensor();
    state.pre_hx_  = trans_res[1].toTensorVector();
    state.pre_cx_  = trans_res[2].toTensorVector();
    state.post_hx_  = trans_res[3].toTensorVector();
    state.post_cx_  = trans_res[4].toTensorVector();
  }

  void greedy_decode (int socket, State& state) {
    auto model = socket_model_[socket];
    // init flags
    auto symbols_added = torch::zeros(state.batch_size_, torch::dtype(torch::kInt32));
    auto time_idx = torch::zeros(state.batch_size_, torch::dtype(torch::kInt32));
    auto fi = state.f_[0];

    while (true) {
      // 2. do prediction
      auto pred_res = model.prediction({state.pre_g_, state.pre_hg_, state.pre_cg_}).toTuple()->elements();
      auto g = pred_res[0].toTensor();
      auto hg = pred_res[1].toTensorVector();
      auto cg = pred_res[2].toTensorVector();
      // 3. do joint
      auto y = model.joint({fi, g[0]}).toTensor();
      auto symbols = torch::argmax(y, 1);
      // 4. update state & flags
      bool finish = model.update({
          symbols, symbols_added, state.res_, state.res_idx_,
          state.f_, state.f_lens_, time_idx, fi,
          state.pre_g_, state.pre_hg_, state.pre_cg_, hg, cg}).toBool();
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
};

}  // namespace models
}  // namespace rnnt
