#include <loadgen.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <thread>
#include <unistd.h>

#include "cxxopts.hpp"
#include "torch_sut.hpp"
#include "test_settings.h"

std::map<std::string, mlperf::TestScenario> scenario_map = {
  {"Offline", mlperf::TestScenario::Offline},
  {"Server", mlperf::TestScenario::Server}};

int main(int argc, char **argv) {
  cxxopts::Options opts (
    "rnnt_inference", "MLPerf Benchmark, RNN-T Inference");
  // opts.allow_unrecognised_options();
  opts.add_options()
    ("m,model_file", "Torch Model File",
     cxxopts::value<std::string>())

    ("s,sample_file", "LibriSpeech Sample File",
     cxxopts::value<std::string>())

    ("k,test_scenario", "Test scenario [Offline, Server]",
     cxxopts::value<std::string>()->default_value("Offline"))

    ("n,inter_parallel", "Instance Number",
     cxxopts::value<int>()->default_value("1"))

    ("j,intra_parallel", "Thread Number Per-Instance",
     cxxopts::value<int>()->default_value("4"))

    ("c,mlperf_config", "Configuration File for LoadGen",
     cxxopts::value<std::string>()->default_value("mlperf.conf"))

    ("u,user_config", "User Configuration for LoadGen",
     cxxopts::value<std::string>()->default_value("user.conf"))

    ("o,output_dir", "Test Output Directory",
     cxxopts::value<std::string>()->default_value("mlperf_output"))

    ("b,batch_size", "Offline Model Batch Size",
     cxxopts::value<int>()->default_value("1"))

    ("perf_count", "Max running sample number",
     cxxopts::value<int>()->default_value("2513"))

    ("disable-hyperthreading", "Whether system enabled hyper-threading or not",
     cxxopts::value<bool>()->default_value("false"))

    ("a,accuracy", "Run test in accuracy mode instead of performance",
     cxxopts::value<bool>()->default_value("false"))

    ("p,profiler", "Whether output trace json or not",
     cxxopts::value<bool>()->default_value("false"))

    ("f,profiler_folder", "If profiler is True, output json in profiler_folder",
     cxxopts::value<std::string>()->default_value("logs"))

    ("preprocessor_file", "Audio Preprocessor File",
     cxxopts::value<std::string>())

    ("preprocessor", "Whether enbale audio preprocess or not",
     cxxopts::value<bool>()->default_value("false"))

    ;

  auto parsed_opts = opts.parse(argc, argv);

  auto model_file = parsed_opts["model_file"].as<std::string>();
  auto sample_file = parsed_opts["sample_file"].as<std::string>();
  auto inter_parallel = parsed_opts["inter_parallel"].as<int>();
  auto intra_parallel = parsed_opts["intra_parallel"].as<int>();
  auto output_dir = parsed_opts["output_dir"].as<std::string>();
  auto mlperf_conf = parsed_opts["mlperf_config"].as<std::string>();
  auto user_conf = parsed_opts["user_config"].as<std::string>();
  auto batch_size = parsed_opts["batch_size"].as<int>();
  auto perf_count = parsed_opts["perf_count"].as<int>();
  auto disable_ht = parsed_opts["disable-hyperthreading"].as<bool>();
  auto test_scenario = parsed_opts["test_scenario"].as<std::string>();
  auto accuracy_mode = parsed_opts["accuracy"].as<bool>();
  auto profiler_flag = parsed_opts["profiler"].as<bool>();
  auto profiler_folder = parsed_opts["profiler_folder"].as<std::string>();
  auto preprocessor_flag = parsed_opts["preprocessor"].as<bool>();
  auto preprocessor_file = parsed_opts["preprocessor_file"].as<std::string>();

  mlperf::TestSettings testSettings;
  mlperf::LogSettings logSettings;
  logSettings.log_output.outdir = output_dir;

  RNNTSUT sut(
    model_file, sample_file, preprocessor_file,
    inter_parallel, intra_parallel,  batch_size, !disable_ht,
    profiler_flag, profiler_folder, preprocessor_flag, test_scenario,
    perf_count);
  
  testSettings.scenario = scenario_map[test_scenario];
  testSettings.FromConfig(mlperf_conf, "rnnt", test_scenario);
  testSettings.FromConfig(user_conf, "rnnt", test_scenario);

  if (accuracy_mode)
    testSettings.mode = mlperf::TestMode::AccuracyOnly;

  std::cout << "Start " << test_scenario << " testing..." << std::endl;
  mlperf::StartTest(&sut, sut.GetQSL(), testSettings, logSettings);
  std::cout << "Testing done." << std::endl;

  return 0;
}
