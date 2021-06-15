#pragma once

#include <opencv2/core/core.hpp>

typedef unsigned char uchar;

struct Image
{
  Image()
    : data(nullptr)
    , rows(0)
    , cols(0)
    , stride(0)
  {
  }

  explicit Image(cv::Mat& cv_mat)
    : data(cv_mat.data)
    , rows(cv_mat.rows / 3)
    , cols(cv_mat.cols)
    , stride(cv_mat.step[0])
  {
    assert(cv_mat.isContinuous());
  }

  uchar *data;
  unsigned rows;
  unsigned cols;
  unsigned stride;
};
