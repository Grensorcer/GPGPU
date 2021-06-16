#include <cstdlib>
#include <vector>
#include <opencv2/core/core.hpp>
#include <benchmark/benchmark.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "lbp.hh"
#include "utils.hh"
#include "image.hh"

typedef cv::Mat_<unsigned char> cvmat;

std::vector<cvmat> data = []()
{
  std::string path = std::getenv("BENCH");
  if (!path.size())
  {
    Log::err("Expected an image\n");
    exit(1);
  }

  cv::VideoCapture capture(path);
  cv::Mat_<unsigned char> full_img;

  if (!capture.isOpened())
  {
    Log::err("Cannot load data in '", path, "' (maybe the file does not exist ?)");
    exit(1);
  }


  capture >> full_img;

  std::vector<cvmat> data;
  if (full_img.rows == 0 && full_img.cols == 0)
  {
    Log::err("Cannot load data in '", path, "' (maybe the file does not exist ?)");
    exit(1);
  }

  data.push_back(full_img);
/*
  for (unsigned i = 1; i < 6; ++i)
  {
    float f = 0.2*i;
    cv::Mat img;
    cv::resize(full_img, img, cv::Size(), f, f, cv::INTER_AREA);
    img.convertTo(img, cv::DataType<unsigned char>::type);
    data.push_back(img);
  }
*/

  return data;
}();

static void warmup(benchmark::State& s)
{
  Image img;

  for (auto _ : s)
  {
    img = Image(data[s.range(0)]);
    benchmark::ClobberMemory();
    benchmark::DoNotOptimize(extract_feature_vector_naive(img.data, img.cols, img.rows));
    
    benchmark::DoNotOptimize(img);
  }

  s.counters["rows"] = img.rows; 
  s.counters["cols"] = img.cols; 
  s.counters["pix"] = img.rows * img.cols; 
}


static void naive(benchmark::State& s)
{
  Image img;

  for (auto _ : s)
  {
    img = Image(data[s.range(0)]);
    benchmark::ClobberMemory();
    unsigned short *res;
    benchmark::DoNotOptimize(res = extract_feature_vector_naive(img.data, img.cols, img.rows));
    free(res);
    
    benchmark::DoNotOptimize(img);
  }

  s.counters["rows"] = img.rows; 
  s.counters["cols"] = img.cols; 
  s.counters["pix"] = img.rows * img.cols; 
}

static void v1(benchmark::State& s)
{
  Image img;

  for (auto _ : s)
  {
    img = Image(data[s.range(0)]);
    benchmark::ClobberMemory();
    unsigned *res;
    benchmark::DoNotOptimize(res = extract_feature_vector_v1(img.data, img.cols, img.rows));
    free(res);
    
    benchmark::DoNotOptimize(img);
  }

  s.counters["rows"] = img.rows; 
  s.counters["cols"] = img.cols; 
  s.counters["pix"] = img.rows * img.cols; 
}


//BENCHMARK(warmup)->DenseRange(0, data.size() - 1);
//BENCHMARK(naive)->DenseRange(0, data.size() - 1);
BENCHMARK(v1)->DenseRange(0, data.size() - 1);
BENCHMARK_MAIN();
