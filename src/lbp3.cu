#include "lbp.hh"
#include "utils.hh"
#include <thread>

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

__global__ void greyscale_v2(uchar* data, size_t width, size_t height, size_t pitch)
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

__global__ void compare_neighbors_v2(uchar* data,
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

__global__ void compute_histograms_by_tiles_v2(uchar* data,
                                            size_t width, size_t height,
                                            size_t pitch,
                                            size_t tile_dim,
                                            char* hists,
                                            size_t h_pitch)
{
  __shared__ unsigned local_h[256];

  // input  data[... + 1] = texton
  // output data[... + 0] = histograms (in each tile)

  size_t x_tile = blockIdx.x;
  size_t y_tile = blockIdx.y;
  size_t x_begin = x_tile * tile_dim;
  size_t y_begin = y_tile * tile_dim;

  if (x_begin >= width || y_begin >= height)
    return;


  //size_t x_end = x_begin + tile_dim;
  //size_t y_end = y_begin + tile_dim;

  unsigned nb_tiles_x = width / 16;


  local_h[threadIdx.y * tile_dim + threadIdx.x] = 0;
  __syncthreads();

  uchar texton = data[(y_begin + threadIdx.y) * pitch + (x_begin + threadIdx.x) * 3 + 1];
  atomicAdd(&local_h[texton], 1);
  __syncthreads();

  short* h = (short*)(hists + (y_tile * nb_tiles_x + x_tile) * h_pitch);
  h[threadIdx.y * tile_dim+ threadIdx.x] =
    local_h[threadIdx.y * tile_dim + threadIdx.x];
}

short* extract_feature_vector_v2(uchar* data, unsigned width, unsigned height,
    short** r_feature_vector, size_t* r_pitch, uchar** gpu_img, size_t* img_pitch)
{
  uchar* d_img;
  size_t pitch;

  checkErr(cudaMallocPitch(&d_img, &pitch, width * 3 * sizeof(uchar), height));

  unsigned nb_tiles_x = width / 16;
  unsigned nb_tiles_y = height / 16;

  char* hists;
  size_t h_pitch;
  checkErr(cudaMallocPitch(&hists, &h_pitch, 256 * sizeof(short), nb_tiles_x
        * nb_tiles_y));

  cudaStream_t stream[4];

  size_t height_of_pipeline = nb_tiles_y / 4 * 16;

  short* h_hists = (short*)malloc(256 * nb_tiles_x
      * nb_tiles_y * sizeof(short));


  auto pipeline = [&](unsigned id)
  {
    uchar* begin_data = data + height_of_pipeline * id * 3 * width * sizeof(uchar);
    uchar* begin_data_dst_copy = d_img + height_of_pipeline * id * pitch;
    if (id != 0)
    {
      begin_data += 3 * width;
      begin_data_dst_copy += pitch;
    }

    size_t height_to_copy = height_of_pipeline;
    if (id == 0)
      ++height_to_copy;
    else if (id == 3)
      --height_to_copy;

    checkErr(cudaMemcpy2DAsync(begin_data_dst_copy, pitch,
                        begin_data, 3 * width * sizeof(uchar),
                        3 * width * sizeof(uchar), height_to_copy,
                        cudaMemcpyHostToDevice, stream[id]));

    uchar* begin_d_img = d_img + height_of_pipeline * id * pitch;

    int bsize = 32;
    int w     = std::ceil((float)width / bsize);
    int h     = std::ceil((float)height / bsize);
    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    Log::dbg("greyscale() + compare_neighbors(): dimGrid: ", w, ' ', h);

    greyscale_v2<<<dimGrid, dimBlock, 0, stream[id]>>>(begin_data_dst_copy, width, height_to_copy, pitch);
    checkErr(cudaStreamSynchronize(stream[id]));

    compare_neighbors_v2<<<dimGrid, dimBlock, 0, stream[id]>>>(begin_d_img, width, height_of_pipeline, pitch);
    checkErr(cudaStreamSynchronize(stream[id]));

    w = std::ceil((float)width  / 16);
    h = std::ceil(((float)height / 16) / 4);
    dimGrid = dim3(w, h);
    dimBlock = dim3(16, 16);
    Log::dbg("compute_histograms_by_tiles(): dimGrid: ", w, ' ', h);

    short* begin_hist = (short*)(hists + (id * height_of_pipeline / 16) * nb_tiles_x * h_pitch);
    compute_histograms_by_tiles_v2<<<dimGrid, dimBlock, 0, stream[id]>>>(begin_d_img, width, height_of_pipeline,
                                                          pitch, 16, (char*)begin_hist, h_pitch);
    checkErr(cudaStreamSynchronize(stream[id]));

    short* begin_h_hist = (short*)(h_hists + (id * nb_tiles_y / 4) * nb_tiles_x * 256);

    Log::dbg(nb_tiles_x * (id * nb_tiles_y / 4));

    checkErr(cudaMemcpy2D(begin_h_hist, 256 * sizeof(short),
                        begin_hist, h_pitch,
                        256*sizeof(short), nb_tiles_x * (nb_tiles_y / 4),
                        cudaMemcpyDeviceToHost));


  };

  std::thread ts[4];
  for (unsigned i = 0; i < 4; ++i)
  {
    checkErr(cudaStreamCreate(&stream[i]));
    ts[i] = std::thread(pipeline, i);
  }

  for (unsigned i = 0; i < 4; ++i)
  {
    ts[i].join();
    checkErr(cudaStreamDestroy(stream[i]));
  }



#if defined(DEBUG)
  // Only keep the texton
  for (size_t y = 0; y < height; ++y)
    for (size_t x = 0; x < width; ++x)
    {
      data[(y * width + x) * 3 + 0] = 0;
      data[(y * width + x) * 3 + 2] = 0;
    }

  checkErr(cudaMemcpy2D(data, width * 3,
                        d_img, pitch,
                        width * 3, height,
                        cudaMemcpyDeviceToHost));

#endif


  if (r_feature_vector)
  {
    *r_feature_vector = (short*)hists;
    *r_pitch = h_pitch;
    *gpu_img = d_img;
    *img_pitch = pitch;
  }
  else {
    checkErr(cudaFree(hists));
    checkErr(cudaFree(d_img));
  }

  return h_hists;
}
