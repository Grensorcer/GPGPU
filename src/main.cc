#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "image.hh"

int main(int argc, char* argv[])
{
  if (argc != 2)
  {
    std::cerr << "Expected an image or a video\n";
    return 1;
  }

  cv::namedWindow("gpgpu", cv::WINDOW_NORMAL);
  cv::resizeWindow("gpgpu", 600,600);

  cv::VideoCapture capture(argv[1]);
  cv::Mat_<unsigned char> frame;

  if (!capture.isOpened())
    throw std::invalid_argument("Error: could not open " + std::string(argv[1]));

  for (capture >> frame; !frame.empty(); capture >> frame)
  {
    Image img(frame);

    for (unsigned i = 0; i < img.rows * 3; i += 3)
      for (unsigned j = 0; j < img.cols * 3; j += 3)
        img.data[i * img.cols + j] = 0;

    cv::imshow("gpgpu", frame);
    cv::resizeWindow ("gpgpu", 400, 200);
    // Display the frame for 20ms.
    cv::waitKey(20);
  }

  cv::waitKey(0);
}
