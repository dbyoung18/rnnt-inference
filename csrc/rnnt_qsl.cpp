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
    const char* feas_name,
    const char* fea_lens_name) {
  auto datasets = GetDictFrom(filename);
  feas_set_ = GetTensorListFrom(datasets, feas_name);
  fea_lens_set_ = GetTensorListFrom(datasets, fea_lens_name);
  CheckSampleCount();
}

RNNTQuerySampleLibrary::RNNTQuerySampleLibrary(
    const std::string& f_feas,
    const std::string& f_fea_lens) {
  feas_set_ = GetTensorListFrom(f_feas);
  fea_lens_set_ = GetTensorListFrom(f_fea_lens);
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
// For length 49 ~ 500, each with a bucket of std::list
//
Queue_t RNNTQuerySampleLibrary::Sort(
    const std::vector<QuerySample>& samples, bool reverse,
    size_t minLength, size_t maxLength) const {
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
    std::vector<QuerySampleIndex> indices) const {
  TensorList feas_list, fea_lens_list;

  feas_list.reserve(indices.size());
  fea_lens_list.reserve(indices.size());

  int64_t maxLength = 0;
 
  for (auto index : indices) {
    auto feas = feas_set_[index];
    auto fea_lens = fea_lens_set_[index];
 
    if (maxLength == 0)
      maxLength = feas.size(0);

    auto len = feas.size(0);
    if (len < maxLength) {  // Padding needed
      std::vector<int64_t> newShape {maxLength};

      auto opts = at::TensorOptions().dtype<int>().memory_format(at::MemoryFormat::Contiguous);

      auto padded_feas = at::zeros(newShape, opts);
      padded_feas.narrow(0, 0, len).copy_(feas);
      feas_list.emplace_back(padded_feas);
    } else {
      feas_list.emplace_back(feas);
    }
     fea_lens_list.emplace_back(fea_lens);
  }
  auto feas = at::stack(feas_list, 0);  // {N, T}
  auto fea_lens = at::cat(fea_lens_list);
  return Stack {feas, fea_lens};
}

}
