#include "metadata.hpp"

namespace rnnt {

State::State (int32_t batch_size, bool enable_bf16) {
  init(batch_size, enable_bf16);
}

// allocation
void State::init (int32_t batch_size, bool enable_bf16) {
  batch_size_ = batch_size;
  enable_bf16_ = enable_bf16;
  // init transcription tensors
  for (int32_t layer = 0; layer < PRE_NUM_LAYERS; ++layer) {
    pre_hx_.emplace_back(torch::empty({batch_size_, TRANS_HIDDEN_SIZE}, torch::kInt8));
    pre_cx_.emplace_back(torch::empty({batch_size_, TRANS_HIDDEN_SIZE}, torch::kFloat16));
  }
  for (int32_t layer = 0; layer < POST_NUM_LAYERS; ++layer) {
    post_hx_.emplace_back(torch::empty({batch_size_, TRANS_HIDDEN_SIZE}, torch::kInt8));
    post_cx_.emplace_back(torch::empty({batch_size_, TRANS_HIDDEN_SIZE}, torch::kFloat16));
  }
  // init prediction tensors
  pre_g_ = torch::full({1, batch_size_}, SOS, torch::kInt32);
  auto pred_dtype = enable_bf16_ ? at::ScalarType::BFloat16 : torch::kFloat32;
  for (int64_t layer = 0; layer < PRED_NUM_LAYERS; ++layer) {
    pre_hg_.emplace_back(torch::empty({batch_size_, PRED_HIDDEN_SIZE}, pred_dtype));
    pre_cg_.emplace_back(torch::empty({batch_size_, PRED_HIDDEN_SIZE}, torch::kFloat32));
  }
  // init res tensors
  res_ = torch::empty({batch_size_, int(HALF_MAX_FEA_LEN * MAX_SYMBOLS_PER_STEP)}, torch::kInt32);
  res_idx_ = torch::empty({batch_size_}, torch::kInt32);
  // init infer index
  finish_idx_ = torch::empty({batch_size_}, torch::kBool);
  finish_size_ = 0;
  remain_lens_ = torch::empty({batch_size_}, torch::kInt32);
  split_idx = 0;
  F_ = torch::empty({MAX_FEA_LEN, batch_size_, TRANS_INPUT_SIZE});
  F_lens_ = torch::empty({batch_size_}, torch::kInt64);
}

// for Offline: batch ahead
void State::update (at::Tensor f, at::Tensor f_lens, int32_t split_len) {
  // update f & f_lens
  if (split_len == -1) {
    f_ = f;
    f_lens_ = f_lens;
    F_lens_ = f_lens;
  } else {
    F_lens_ = f_lens;
    // f = at::pad(f, {0, 0, 0, 0, 0, MAX_FEA_LEN-f.size(0)}, "constant", 0);
    f_split_ = torch::split(f, split_len);
    split_lens_ = at::full({batch_size_}, split_len, torch::kInt32);
    remain_lens_ = f_lens.clone();
  }
}

// for Server: dynamic batch
void State::update (TensorVector f, TensorVector f_lens, int32_t split_len) {
  int j = 0;
  for (int32_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    if (finish_idx_[batch_idx].item().toBool() == true) {
      auto f_len = f_lens[j].item().toInt();
      F_.index_put_({Slice(0, f_len), batch_idx}, f[j].index({Slice(0, f_len)}).squeeze());
      F_lens_.index_put_({batch_idx}, f_len);
      ++j;
    }
  }
  if (split_len != -1)
    split_lens_ = at::full({batch_size_}, split_len, torch::kInt32);
  f_ = F_;
  f_lens_ = F_lens_;
}

bool State::next () {
  bool status = (split_idx != f_split_.size() && finish_size_ != batch_size_);
  if (status) {
    f_lens_ = torch::min(split_lens_, remain_lens_);
    f_ = f_split_[split_idx];
    split_idx++;
    remain_lens_ -= f_lens_;
    finish_idx_ = f_lens_.eq(0);
    finish_size_ = finish_idx_.count_nonzero().item().toInt();
    // status &= (finish_size_ != batch_size_);
  }
  return status;
}

// bool State::next (int32_t split_len, int32_t count) {
//   reset(F_lens_.size(0), split_len);
//   f_lens_ = torch::min(split_lens_, (F_lens_ - remain_lens_).clamp(0));
//   f_ = F_.index({Slice(split_idx, split_idx + split_len, None)});
//   finish_idx_ = f_lens_.eq(0);
//   finish_size_ = torch::count_nonzero(finish_idx_).item().toInt();
//   split_idx += split_len;
//   return (finish_size_ != batch_size_);
// }

void State::reset (int batch_size) {
  // checking actual BS
  if (batch_size != batch_size_)
    init(batch_size, enable_bf16_);
  for (int32_t layer = 0; layer < PRE_NUM_LAYERS; ++layer) {
    pre_hx_[layer].zero_();
    pre_cx_[layer].zero_();
  }
  for (int32_t layer = 0; layer < POST_NUM_LAYERS; ++layer) {
    post_hx_[layer].zero_();
    post_cx_[layer].zero_();
  }
  // init prediction tensors
  pre_g_.fill_(SOS);
  for (int64_t layer = 0; layer < PRED_NUM_LAYERS; ++layer) {
    pre_hg_[layer].zero_();
    pre_cg_[layer].zero_();
  }
  // init res tensors
  res_.fill_(SOS);
  res_idx_.fill_(-1);
  // init infer index
  finish_idx_.fill_(true);
  finish_size_ = 0;
  remain_lens_.zero_();
  split_idx = 0;
  F_.zero_();
  F_lens_.zero_();
}

// TODO: reset split_idx
void State::reset (int32_t batch_size, int32_t split_len) {
  f_ = torch::zeros({split_len, batch_size, TRANS_INPUT_SIZE});
  f_lens_ = torch::zeros({batch_size}, torch::kInt32);
  // if (split_len != split_lens_[0].item().toInt())
    // split_lens_ = at::full({batch_size_}, split_len, torch::kInt32);
  if (batch_size != batch_size_)
    init(batch_size);
}

}  // namespace rnnt
