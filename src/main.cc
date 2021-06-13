#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "image.hh"
#include "utils.hh"

#include "lbp.hh"

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
    unsigned short* hists = extract_feature_vector(img.data, img.rows, img.cols);

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

      for (unsigned i = 0; i < 50*50; ++i)
      {
        unsigned short *h = hists + 256 * i;

        for (unsigned j = 0; j < 256; j++)
          f << h[j] << ",\n"; 
      }

      /*
      for (size_t y_begin = 0; y_begin < img.rows; y_begin += 16)
      {
        size_t y_end = y_begin + 16;

        if (y_end > img.rows)
          continue;

        for (size_t x_begin = 0; x_begin < img.cols; x_begin += 16)
        {
          size_t x_end = x_begin + 16;
          if (x_end > img.cols)
            continue;

          for (size_t y = y_begin; y < y_end; ++y)
            for (size_t x = x_begin; x < x_end; ++x)
              f << std::to_string(img.data[(y * img.cols + x) * 3]) << ",\n"; 
        }
      }
      */

      f.close();

      Log::dbg(img.rows / 16, ' ', img.cols / 16);

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
