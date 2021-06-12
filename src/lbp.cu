#include "lbp.hh"
#include "utils.hh"

#if defined(DEBUG)
  [[gnu::noinline]]
  void abortOnError(cudaError_t err,
                    const char* fname, int line,
                    bool abort_on_error=true)
  {
    if (err)
    {
      cudaError_t err = cudaGetLastError();
      Log::err("CudaError at line ", line, " in ", fname, ":\n"
                "        ", cudaGetErrorName(err), ": ", cudaGetErrorString(err)); 

      if (abort_on_error)
        std::exit(1);
    }
  }

  #define checkErr(err) { abortOnError((err), __FILE__, __LINE__); }

  #define checkKernel() { \
                        checkErr(cudaPeekAtLastError()); \
                        checkErr(cudaDeviceSynchronize()); \
                        }
#else
  #define checkErr(err)
  #define checkKernel()
#endif

__global__ void blue(uchar* img, size_t width, size_t height, size_t pitch)
{
  size_t x = blockDim.x * blockIdx.x + threadIdx.x;
  size_t y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  img[y * pitch + x * 3] = 255;
  img[y * pitch + x * 3 + 1] = 0;
  img[y * pitch + x * 3 + 2] = 0;
}

void extract_feature_vector(uchar *data, unsigned width, unsigned height)
{
  uchar* d_img;
  size_t pitch;
  checkErr(cudaMallocPitch(&d_img, &pitch, width * 3 * sizeof(uchar), height));


  checkErr(cudaMemcpy2D(d_img, pitch,
                        data, width * 3,
                        width * 3, height,
                        cudaMemcpyHostToDevice));

  {
    int bsize = 32;
    int w     = std::ceil((float)width / bsize);
    int h     = std::ceil((float)height / bsize);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    blue<<<dimGrid, dimBlock>>>(d_img, width, height, pitch);
    checkKernel();
  }

  checkErr(cudaMemcpy2D(data, width * 3,
                        d_img, pitch,
                        width * 3, height,
                        cudaMemcpyDeviceToHost));

  checkErr(cudaFree(d_img));
}
