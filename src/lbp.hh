#pragma once

typedef unsigned char uchar;

unsigned short* extract_feature_vector_naive(uchar *data, unsigned cols, unsigned rows);
unsigned* extract_feature_vector_v1(uchar *data, unsigned cols, unsigned rows);
short* extract_feature_vector_v2(uchar *data, unsigned cols, unsigned rows,
    short** r_feature_vector=nullptr, size_t* r_pitch=nullptr,
    uchar** gpu_img=nullptr, size_t* img_pitch=nullptr);
