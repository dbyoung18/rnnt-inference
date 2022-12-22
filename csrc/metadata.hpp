#pragma once
#include <ATen/core/ivalue.h>
#include <string>
#include <torch/script.h>
#include <ATen/Parallel.h>

namespace rnnt {
using TensorVector = std::vector<at::Tensor>;
using namespace torch::indexing;

static const std::vector<char> LABELS = {
  ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\''};

enum {
  PRE_NUM_LAYERS = 2,
  POST_NUM_LAYERS = 3,
  TRANS_INPUT_SIZE = 240,
  TRANS_HIDDEN_SIZE = 1024,
  PRED_NUM_LAYERS = 2,
  PRED_HIDDEN_SIZE = 320,
  JOINT_HIDDEN_SIZE = 512,
  SOS = -1,
  BLANK = 28,
  MAX_SYMBOLS_PER_STEP = 30,
  MAX_LEN = 500,
  HALF_MAX_LEN = 250
};

// batch samples
class State {

public:

  State();

  State(int32_t batch_size, bool enable_bf16 = true);

  virtual ~State() = default;

  void init(int32_t batch_size, bool enable_bf16 = true);
  void update(at::Tensor x, at::Tensor x_lens, int32_t split_len = -1);
  void update (TensorVector x, TensorVector x_lens, int32_t split_len);
  bool next ();
  void reset ();
  void reset (int32_t batch_size, int32_t split_len);

  int32_t batch_size_;
  int32_t finish_size_;
  at::Tensor split_lens_;
  bool enable_bf16_ = true;
  // transcription
  at::Tensor F_;
  at::Tensor F_lens_;
  TensorVector f_split_;
  TensorVector f_lens_split_;
  at::Tensor f_;
  at::Tensor f_lens_;
  TensorVector pre_hx_;
  TensorVector pre_cx_;
  TensorVector post_hx_;
  TensorVector post_cx_;
  // prediction
  at::Tensor pre_g_;
  TensorVector pre_hg_;
  TensorVector pre_cg_;
  // results
  at::Tensor res_;
  at::Tensor res_idx_;
  // infer index
  at::Tensor finish_idx_;  // row
  at::Tensor remain_lens_;  // col
  int32_t split_idx = 0;
};

}  // namespace rnnt
