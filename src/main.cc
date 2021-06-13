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
    unsigned short* hists = extract_feature_vector(img.data, img.cols, img.rows);

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

      unsigned nb_tiles_x = img.cols / 16;
      unsigned nb_tiles_y = img.rows / 16;

      for (unsigned i = 0; i < nb_tiles_x * nb_tiles_y; ++i)
      {
        unsigned short *h = hists + 256 * i;

        for (unsigned j = 0; j < 256; j++)
          f << h[j] << ",\n"; 
      }

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
