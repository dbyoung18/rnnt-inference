#include <c10/core/MemoryFormat.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <ATen/ATen.h>
#include <cassert>
#include "rnnt_qsl.hpp"

namespace qsl {
TensorList RNNTQuerySampleLibrary::GetTensorListFrom(
    at::IValue value) {
  std::vector<at::Tensor> tensor_list;
  auto toTensor = [](at::IValue item) {return item.toTensor();};

  if (value.isList()) {
    auto value_list = value.toList();
    std::transform(value_list.begin(), value_list.end(),
        std::back_inserter(tensor_list), toTensor);
  } else if (value.isTensorList()) {
    auto c10_list = value.toTensorList();
    tensor_list.insert(tensor_list.begin(), c10_list.begin(), c10_list.end());
  } else if (value.isTuple()) {
    auto value_list = value.toTuple()->elements();
    std::transform(value_list.begin(), value_list.end(),
        std::back_inserter(tensor_list), toTensor);
  } else {
    TORCH_CHECK(false, "Can't get TensorList from IValue type: ", value.tagKind());
  }

  return tensor_list;
}

TensorList RNNTQuerySampleLibrary::GetTensorListFrom(
    const std::string& filename) {
  caffe2::serialize::PyTorchStreamReader reader(filename);
  auto stack = torch::jit::readArchiveAndTensors("data",
      "","",
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      reader);
  return GetTensorListFrom(stack);
}

TensorList RNNTQuerySampleLibrary::GetTensorListFrom(
    c10::Dict<at::IValue, at::IValue>& dict,
    const char* name) {
  at::IValue ivname (name);

  auto tensor_list = dict.find(ivname);
  if ( tensor_list != dict.end() )
    return GetTensorListFrom(tensor_list->value());
  else
    return TensorList();
}

Stack RNNTQuerySampleLibrary::GetIValueListFrom(at::IValue value) {
  auto tensor_list = GetTensorListFrom(value);
  Stack stack = {tensor_list[0], tensor_list[1]};
  return stack;
}

c10::Dict<at::IValue, at::IValue> RNNTQuerySampleLibrary::GetDictFrom(
    const std::string& filename) {
  caffe2::serialize::PyTorchStreamReader reader(filename);
  auto stack = torch::jit::readArchiveAndTensors("data",
      "","",
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      reader);
  // Exception management
  return stack.toGenericDict();
}

RNNTQuerySampleLibrary::RNNTQuerySampleLibrary(
    const std::string& filename,
    const char* x_name,
    const char* x_lens_name) {
  auto datasets = GetDictFrom(filename);
  x_set_ = GetTensorListFrom(datasets, x_name);
  x_lens_set_ = GetTensorListFrom(datasets, x_lens_name);
  minLength = at::min(at::cat(x_lens_set_)).item().toInt();
  maxLength = at::max(at::cat(x_lens_set_)).item().toInt();
  CheckSampleCount();
}

RNNTQuerySampleLibrary::RNNTQuerySampleLibrary(
    const std::string& f_x,
    const std::string& f_x_lens) {
  x_set_ = GetTensorListFrom(f_x);
  x_lens_set_ = GetTensorListFrom(f_x_lens);
  minLength = at::min(at::cat(x_lens_set_)).item().toInt();
  maxLength = at::max(at::cat(x_lens_set_)).item().toInt();
  CheckSampleCount();
}

RNNTQuerySampleLibrary RNNTQuerySampleLibrary::Create(
    const std::string& filename) {
  return RNNTQuerySampleLibrary(filename);
}

void RNNTQuerySampleLibrary::CheckSampleCount() {
  /* throw if two sets have different sizes */
  std::cout << "Load " << PerformanceSampleCount() << " samples" << std::endl;
}

//
// Parallel bucket sort (unstable) would be the most efficient choice
// For length minLength ~ maxLength, each with a bucket of std::list
//
Queue_t RNNTQuerySampleLibrary::Sort(
    const std::vector<QuerySample>& samples, bool preprocessor,
    bool reverse) const {
  const auto lengthOffset = minLength;
  const auto nBucket = maxLength - lengthOffset + 1;

  std::vector<Queue_t> Buckets(nBucket);
  std::vector<std::mutex> lks(nBucket);

  // (Parallel) sort
  // TODO: support other parallel library
# pragma omp parallel for
  for (const auto &sample : samples) {
    auto length = GetFeatureLength(sample.index);

    auto idx = reverse ? maxLength - length : length - lengthOffset;
    auto& bucket = Buckets[idx];
    auto& l = lks[idx];

    {
      std::unique_lock<std::mutex> guard(l);
      bucket.emplace_back(sample);
    }
  }

  // Splice them together
  Queue_t result;
  for (auto &q : Buckets)
    result.splice(result.end(), std::move(q));

  return result;
}

//
// Assemble samples into larger batch
//
Stack RNNTQuerySampleLibrary::AssembleSamples(
    std::vector<QuerySampleIndex> indices, bool preprocessor) const {
  TensorList x_list, x_lens_list;

  x_list.reserve(indices.size());
  x_lens_list.reserve(indices.size());

  int64_t maxLength = 0;
 
  for (auto index : indices) {
    auto x = x_set_[index];
    auto x_lens = x_lens_set_[index];
 
    if (maxLength == 0)
      maxLength = x.size(0);

    auto len = x.size(0);
    if (len < maxLength) {  // Padding needed
      std::vector<int64_t> newShape;
      if (preprocessor)
        newShape = {maxLength};
      else
        newShape = {maxLength, x.size(1)};

      auto opts = at::TensorOptions().dtype<float>().memory_format(at::MemoryFormat::Contiguous);

      auto padded_x = at::zeros(newShape, opts);
      padded_x.narrow(0, 0, len).copy_(x);
      x_list.emplace_back(padded_x);
    } else {
      x_list.emplace_back(x);
    }
    x_lens_list.emplace_back(x_lens);
  }
  auto x = preprocessor ? at::stack(x_list, 0) : at::stack(x_list, 1);  // {N, T} or {T, N, C}
  auto x_lens = at::cat(x_lens_list);
  return Stack {x, x_lens};
}

}
