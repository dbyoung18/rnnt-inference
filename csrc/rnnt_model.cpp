#include "rnnt_model.hpp"

namespace models {

// template <typename... Args>
// at::IValue TorchModel::inference(Args&&... args) {
  // return greedy_decode(model_, std::forward<Args>(args)...);
// }

// template <typename... Args>
// at::IValue TorchModel::inference_at(int socket, Args&&... args) {
  // return greedy_decode(socket_model_[socket], std::forward<Args>(args)...);
// }

// template <typename... Args>
// at::IValue greedy_decode(torch::jit::script::Module model, Args&&... args) {
  // auto sub_modules = model.children();
  // auto transcription = *sub_modules.begin();
  // return transcription.forward(std::forward<Args>(args)...);
// }

}
