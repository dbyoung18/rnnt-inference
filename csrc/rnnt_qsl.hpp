#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <list>
#include <torch/csrc/jit/serialization/import_read.h>
#include <query_sample_library.h>

namespace qsl {
using namespace mlperf;
using TensorList = std::vector<at::Tensor>;
using Stack = std::vector<at::IValue>;
using Queue_t = std::list<mlperf::QuerySample>;

class RNNTQuerySampleLibrary : public QuerySampleLibrary {
public:
  RNNTQuerySampleLibrary(
      const std::string& filename,
      const char* feas_name = "feas",
      const char* fea_lens_name = "fea_lens");

  RNNTQuerySampleLibrary(
      const std::string& f_feas,
      const std::string& f_fea_lens);

  virtual ~RNNTQuerySampleLibrary() = default;

  const std::string& Name() const override {
    static const std::string name("RNN-T LibriSpeech QSL");
    return name;
  }

  size_t TotalSampleCount() override {
    return feas_set_.size();
  }

  void CheckSampleCount();

  size_t PerformanceSampleCount() override {
    return TotalSampleCount();
  }

  // LibriSpeech is small enough to be in Memory
  void LoadSamplesToRam(const std::vector<QuerySampleIndex>& samples) override {}
  void UnloadSamplesFromRam(
      const std::vector<QuerySampleIndex>& samples) override {}

  static RNNTQuerySampleLibrary Create(const std::string& filename);

  Stack GetSample(QuerySampleIndex index) const {
    return {
      feas_set_.at(index),
      fea_lens_set_.at(index),
    };
  }

  Stack AssembleSamples(std::vector<QuerySampleIndex> indices) const;

  // List of tensor of 1d
  size_t GetFeatureLength(size_t index) const {
    return feas_set_[index].size(0);
  }

  // Sort LibriSpeech data for batching
  Queue_t Sort(
      const std::vector<QuerySample>& samples, bool reverse = true,
      size_t minLength=49, size_t maxLength=500) const;

  c10::Dict<at::IValue, at::IValue> GetDictFrom(const std::string& filename);

  static TensorList GetTensorListFrom(at::IValue value);
  static TensorList GetTensorListFrom(const std::string& filename);
  static TensorList GetTensorListFrom(
      c10::Dict<at::IValue, at::IValue>& dict,
      const char* name);

private:
  TensorList feas_set_;
  TensorList fea_lens_set_;
};

}
