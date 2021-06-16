#pragma once

typedef unsigned char uchar;

unsigned short* extract_feature_vector_naive(uchar *data, unsigned cols, unsigned rows);
unsigned* extract_feature_vector_v1(uchar *data, unsigned cols, unsigned rows);
