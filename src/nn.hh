#pragma once

#if defined(DEBUG)

  // Defined in lbp.cu
  [[gnu::noinline]]
  void abortOnError(cudaError_t err,
                    const char* fname, int line,
                    bool abort_on_error=true);

  #define checkErr(cudaCall) { abortOnError((cudaCall), __FILE__, __LINE__); }

  #define checkKernel() {                                     \
                          checkErr(cudaPeekAtLastError());    \
                          checkErr(cudaDeviceSynchronize());  \
                        }
#else
  #define checkErr(cudaCall) cudaCall
  #define checkKernel()
#endif

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

int step_2(rtype hists_cpu, uint8_t* image, size_t width, size_t height);
rtype read_hist_csv();