#include "lbp.hh"
#include "utils.hh"

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

__global__ void greyscale_opti(uchar* data, size_t width, size_t height, size_t pitch)
{
  // input  data: image bgr
  // otuput data[... + 0] image grey 
  size_t x = blockDim.x * blockIdx.x + threadIdx.x;
  size_t y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  uchar* pix = &data[y * pitch + x * 3];
  uchar grey = pix[0] / 3 + pix[1] / 3 + pix[2] / 3;

  pix[0] = grey;
}

__global__ void compare_neighbors_opti(uchar* data,
                                  size_t width, size_t height, size_t pitch)
{
  // input  data[... + 0] image grey
  // output data[... + 1] texton
  size_t x = blockDim.x * blockIdx.x + threadIdx.x;
  size_t y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  uchar* pix = &data[y * pitch + x * 3];
  uchar* res = &pix[1];
  uchar c = pix[0];

  (*res) = 0;

#pragma unroll
  for (size_t j = 0; j < 3; ++j)
    for (size_t i = 0; i < 3; ++i)
      if (
          // Checks the boundary, consider the pixels outside of the image as 0
          (x + i - 1) < width && (y + j - 1) < height 
          // Do not compare the pix with itself
          && (i != 1 && j != 1)
          //Finally, compare the pixel with its neighbor
          && long(data[(y + j - 1ULL) * pitch + (x + i - 1ULL) * 3ULL])
          - long(c) >= 0
          )
      {
        // The results of the compraison is store in a single bit
        unsigned idx = i * 3U + j;
        if (idx > 4)
          --idx;
        (*res) |= 1U << idx;
      }
}

__global__ void compute_histograms_by_tiles_opti(uchar* data,
                                            size_t width, size_t height,
                                            size_t pitch,
                                            size_t tile_dim,
                                            char* hists,
                                            size_t h_pitch)
{
  // input  data[... + 1] = texton
  // output data[... + 0] = histograms (in each tile)

  size_t x_tile = blockDim.x * blockIdx.x;
  size_t y_tile = blockDim.y * blockIdx.y;
  size_t x_begin = x_tile * tile_dim;
  size_t y_begin = y_tile * tile_dim;

  if (x_begin >= width || y_begin >= height)
    return;

  //size_t x_end = x_begin + tile_dim;
  size_t y_end = y_begin + tile_dim;

  unsigned nb_tiles_x = width / 16;

  unsigned* h = (unsigned *)(hists + (y_tile * nb_tiles_x + x_tile) * h_pitch);

  for (unsigned i = 0; i < 256; i++)
    h[i] = 0;

  __syncthreads();

  for (size_t y = y_begin; y < y_end && y < height; y += 2)
    {
      uchar texton = data[(y + threadIdx.y) * pitch + threadIdx.x * 3 + 1];
      atomicAdd(&h[texton], 1);
    }
}

unsigned * extract_feature_vector_v1(uchar* data, unsigned width, unsigned height)
{
  uchar* d_img;
  size_t pitch;

  checkErr(cudaMallocPitch(&d_img, &pitch, width * 3 * sizeof(uchar), height));

  checkErr(cudaMemcpy2D(d_img, pitch,
                        data, 3 * width * sizeof(uchar),
                        3 * width * sizeof(uchar), height,
                        cudaMemcpyHostToDevice));

  int bsize = 32;
  int w     = std::ceil((float)width / bsize);
  int h     = std::ceil((float)height / bsize);

  dim3 dimBlock(bsize, bsize);
  dim3 dimGrid(w, h);

  Log::dbg("greyscale() + compare_neighbors(): dimGrid: ", w, ' ', h);

  greyscale_opti<<<dimGrid, dimBlock>>>(d_img, width, height, pitch);
  checkKernel();
  cudaDeviceSynchronize();

  compare_neighbors_opti<<<dimGrid, dimBlock>>>(d_img, width, height, pitch);
  checkKernel();
  cudaDeviceSynchronize();

  w = std::ceil((float)width  / 16);
  h = std::ceil((float)height / 16);

  unsigned nb_tiles_x = width / 16;
  unsigned nb_tiles_y = height / 16;

  char* hists;
  size_t h_pitch;
  checkErr(cudaMallocPitch(&hists, &h_pitch, 256 * sizeof(unsigned), nb_tiles_x
        * nb_tiles_y));

  dimGrid = dim3(w, h);
  Log::dbg("compute_histograms_by_tiles(): dimGrid: ", w, ' ', h);
  compute_histograms_by_tiles_opti<<<dimGrid, dimBlock>>>(d_img, width, height,
      pitch, 16, hists, h_pitch);

  checkKernel();
  cudaDeviceSynchronize();

/*
  checkErr(cudaMemcpy2D(data, width * 3,
                        d_img, pitch,
                        width * 3, height,
                        cudaMemcpyDeviceToHost));
*/
#if defined(DEBUG)
  // Only keep the texton
  for (size_t y = 0; y < height; ++y)
    for (size_t x = 0; x < width; ++x)
    {
      data[(y * width + x) * 3 + 0] = 0;
      data[(y * width + x) * 3 + 2] = 0;
    }
#endif

  unsigned * h_hists = (unsigned *)malloc(256 * nb_tiles_x
      * nb_tiles_y * sizeof(unsigned));
  checkErr(cudaMemcpy2D(h_hists, 256 * sizeof(unsigned),
                        hists, h_pitch,
                        256*sizeof(unsigned), nb_tiles_x * nb_tiles_y,
                        cudaMemcpyDeviceToHost));

  checkErr(cudaFree(d_img));

  return h_hists;
}
