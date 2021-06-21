#include <cstdlib>
#include <vector>
#include <opencv2/core/core.hpp>
#include <benchmark/benchmark.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <wait.h>
#include <unistd.h>

#include <string>
#include <fstream>

#include "lbp.hh"
#include "utils.hh"
#include "image.hh"
#include "nn.hh"

typedef cv::Mat_<unsigned char> cvmat;

std::vector<short*> r_feature_vector_v;
std::vector<size_t> r_pitch_v;
std::vector<uchar*> gpu_img_v;
std::vector<size_t> img_pitch_v;

auto data = []()
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
  for (const auto& img : data)
  {
    short* feature_vector;
    uchar* gpu_img;
    size_t r_pitch, img_pitch;
    free(extract_feature_vector_v2(img.data, img.cols, img.rows,
                                    &feature_vector, &r_pitch,
                                    &gpu_img, &img_pitch));
    r_feature_vector_v.push_back(feature_vector);
    r_pitch_v.push_back(r_pitch);
    gpu_img_v.push_back(gpu_img);
    img_pitch_v.push_back(img_pitch);
  }

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

static void v2(benchmark::State& s)
{
  Image img;

  for (auto _ : s)
  {
    img = Image(data[s.range(0)]);
    benchmark::ClobberMemory();
    short *res;
    benchmark::DoNotOptimize(res = extract_feature_vector_v2(img.data, img.cols, img.rows));
    free(res);

    benchmark::DoNotOptimize(img);
  }

  s.counters["rows"] = img.rows; 
  s.counters["cols"] = img.cols; 
  s.counters["pix"] = img.rows * img.cols; 
}

static void bench_step2(benchmark::State& s)
{
  Image img;
  rtype hists = read_hist_csv();

  for (auto _ : s)
  {
    img = Image(data[s.range(0)]);
    benchmark::ClobberMemory();
    int res;
    benchmark::DoNotOptimize(res = step_2(hists, img.data, img.cols, img.rows, "release/cluster.csv"));

    benchmark::DoNotOptimize(img);
  }

  s.counters["rows"] = img.rows; 
  s.counters["cols"] = img.cols; 
  s.counters["pix"] = img.rows * img.cols; 
}

static void bench_step2_v1(benchmark::State& s)
{
  Image img;
  size_t i = 0;

  for (auto _ : s)
  {
    short* r_feature_vector = r_feature_vector_v[i];
    size_t r_pitch = r_pitch_v[i];
    uchar* gpu_img = gpu_img_v[i];
    size_t img_pitch = img_pitch_v[i];

    img = Image(data[s.range(0)]);
    benchmark::ClobberMemory();
    int res2;
    benchmark::DoNotOptimize(res2 = step_2_v1(img.data, img.cols, img.rows, r_feature_vector, r_pitch, gpu_img, img_pitch, "release/cluster.csv"));
    benchmark::DoNotOptimize(img);

    i++;
  }

  s.counters["rows"] = img.rows; 
  s.counters["cols"] = img.cols; 
  s.counters["pix"] = img.rows * img.cols; 
}

static void global(benchmark::State& s)
{
  Image img;

  for (auto _ : s)
  {
    short* r_feature_vector;
    size_t r_pitch;
    uchar* gpu_img;
    size_t img_pitch;

    img = Image(data[s.range(0)]);
    unsigned nb_tiles_x = img.cols / 16;
    unsigned nb_tiles_y = img.rows / 16;

    benchmark::ClobberMemory();
    short *hists;
    benchmark::DoNotOptimize(hists = extract_feature_vector_v2(img.data, img.cols, img.rows, &r_feature_vector, &r_pitch, &gpu_img, &img_pitch));
    free(hists);

    std::fstream f("out.csv", std::fstream::out);
    f << "val,\n";

    for (unsigned i = 0; i < nb_tiles_x * nb_tiles_y; ++i)
    {
      rtype h = hists + 256 * i;

      for (unsigned j = 0; j < 256; j++)
        f << h[j] << ",\n"; 
    }

    f.close();

    // Computes the centroids with python script
    pid_t pid = fork();

    if (pid == 0) {
      char* command = "python";
      char* argument_list[] = {"python", "src/kmeans.py", NULL};
      int status_code = execvp(command, argument_list);

      if (status_code == -1) {
        Log::dbg("Execvp failed");
      }
    }
    else {
        waitpid(pid, NULL, WUNTRACED | WCONTINUED);
    }

    int res;
    benchmark::DoNotOptimize(res = step_2_v1(img.data, img.cols, img.rows, r_feature_vector, r_pitch, gpu_img, img_pitch, "cluster.csv"));

    benchmark::DoNotOptimize(img);
  }

  s.counters["rows"] = img.rows; 
  s.counters["cols"] = img.cols; 
  s.counters["pix"] = img.rows * img.cols; 
}

BENCHMARK(warmup)->DenseRange(0, data.size() - 1);
BENCHMARK(naive)->DenseRange(0, data.size() - 1);
BENCHMARK(v1)->DenseRange(0, data.size() - 1);
BENCHMARK(v2)->DenseRange(0, data.size() - 1);
BENCHMARK(bench_step2)->DenseRange(0, 0/*, data.size() - 1*/); // Cannot bench this one with other file than barcode-00-01.jpg histograms for now
BENCHMARK(bench_step2_v1)->DenseRange(0, data.size() - 1);
BENCHMARK(global)->DenseRange(0, data.size() - 1);
BENCHMARK_MAIN();
