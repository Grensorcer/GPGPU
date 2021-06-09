#include <cstdlib>
#include <vector>
#include <opencv2/core/core.hpp>
#include <benchmark/benchmark.h>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

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
  if (!capture.isOpened())
    throw "Error when reading steam_avi";

  cvmat full_img;
  capture >> full_img;

  std::vector<cvmat> data;

  for (unsigned i = 1; i < 4; ++i)
  {
    float f = 0.2*i;
    cv::Mat img;
    cv::resize(full_img, img, cv::Size(), f, f, cv::INTER_AREA);
    img.convertTo(img, cv::DataType<unsigned char>::type);
    data.push_back(img);
  }

  data.push_back(full_img);
  return data;
}();

static void cpu(benchmark::State& s)
{
  Image img;

  for (auto _ : s)
  {
    img = Image(data[s.range(0)]);
    benchmark::ClobberMemory();
    // TODO add func
    benchmark::DoNotOptimize(img);
  }

  s.counters["rows"] = img.rows; 
  s.counters["cols"] = img.cols; 
  s.counters["pix"] = img.rows * img.cols; 
}

BENCHMARK(cpu)->DenseRange(0, data.size() - 1);
BENCHMARK_MAIN();
