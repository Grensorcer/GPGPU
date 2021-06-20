#pragma once


#if defined(naive)
    typedef unsigned short* rtype;
    typedef unsigned short dtype;
#elif defined(v1)
    typedef unsigned * rtype;
    typedef unsigned dtype;
#else
    typedef short * rtype;
    typedef short dtype;
#endif

typedef unsigned char uchar;

int step_2(rtype hists_cpu, uint8_t* image, size_t width, size_t height, char* cluster_file);
int step_2_v1(uchar* cpu_img, size_t width, size_t height, short* r_feature_vector, size_t r_pitch, uchar* gpu_img, size_t img_pitch, char* cluster_file);
rtype read_hist_csv(); // in nn.cu