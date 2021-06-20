#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <wait.h>
#include <unistd.h>

#include "image.hh"
#include "utils.hh"

#include "lbp.hh"
#include "nn.hh"

#include <string>
#include <fstream>

// This leak 24 bytes.
int main(int argc, char* argv[])
{
  bool display_image = false;

  if (argc != 2)
  {
    std::cerr << "Expected an image or a video\n";
    return 1;
  }

  if (display_image)
  {
    cv::namedWindow("gpgpu", cv::WINDOW_NORMAL);
    cv::resizeWindow("gpgpu", 600,600);
  }

  cv::Mat_<unsigned char> frame;
  cv::VideoCapture capture(argv[1]);

  if (!capture.isOpened())
    throw std::invalid_argument("Error: could not open " + std::string(argv[1]));

  for (capture >> frame; !frame.empty(); capture >> frame)
  {
    Image img(frame);

    unsigned nb_tiles_x = img.cols / 16;
    unsigned nb_tiles_y = img.rows / 16;


    short* r_feature_vector;
    size_t r_pitch;
    uchar* gpu_img;
    size_t img_pitch;

#if defined(naive)
    typedef unsigned short* rtype;
    rtype hists = extract_feature_vector_naive(img.data, img.cols, img.rows);
#elif defined(v1)
    typedef unsigned * rtype;
    rtype hists = extract_feature_vector_v1(img.data, img.cols, img.rows);
#else
    typedef short * rtype;
    rtype hists = extract_feature_vector_v2(img.data, img.cols, img.rows, &r_feature_vector, &r_pitch, &gpu_img, &img_pitch);
#endif

    if (display_image)
    {
      cv::imshow("gpgpu", frame);
      cv::resizeWindow ("gpgpu", 400, 200);

      // Display the frame for 20ms.
      cv::waitKey(20);
    }
    else
    {
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
        char* argument_list[] = {"python", "../src/kmeans.py", NULL};
        int status_code = execvp(command, argument_list);

        if (status_code == -1) {
          Log::dbg("Execvp failed");
          return 1;
        }
      }
      else {
          waitpid(pid, NULL, WUNTRACED | WCONTINUED);
      }

      // Step 2
      //step_2(img.data, img.cols, img.rows);
      step_2_v1(img.data, img.cols, img.rows, r_feature_vector, r_pitch, gpu_img, img_pitch);

      Log::dbg(img.rows / 16, ' ', img.cols/ 16);

      cv::imwrite("out.jpg", frame);
      return 0;
    }
  }

    if (display_image)
  {
    cv::waitKey(0);
    cv::destroyAllWindows();
  }
}
