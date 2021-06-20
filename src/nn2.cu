#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <curand_kernel.h>

#include "utils.hh"
#include "nn.hh"


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

// calculate the Euclidean distance between two vectors
__device__ float euclidean_distance_v1(float *vect1, rtype vect2, int len_vect){
	float distance = 0.0;
	for (size_t i = 0; i < len_vect; i++) 
		distance += (vect1[i] - vect2[i]) * (vect1[i] - vect2[i]);
	return distance;
}

// Locate the most similar neighbors
__global__ void get_neighbor_v1(float *clusters, size_t cluster_pitch, size_t len_clusters, size_t len_vect,
                            rtype patches, size_t patches_pitch, size_t n_patches, uchar *neighbor) {
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x >= n_patches)
        return;

    float distance_mini = -1;
    uchar mini = 0;

    rtype patch = (rtype) ((char*) patches + x * patches_pitch);

	for (size_t i = 0; i < len_clusters; i++) {
        float* cluster = (float*) ((char*) clusters + i * cluster_pitch);

		float dist = euclidean_distance_v1(cluster, patch, len_vect);
        if (dist < distance_mini || distance_mini == -1) {
            distance_mini = dist;
            mini = i;
        }
    }
    neighbor[x] = mini;
}

std::string readFileIntoString_v1(const std::string& path) {
    auto ss = std::ostringstream();
    std::ifstream input_file(path);
    if (!input_file.is_open()) {
        std::cerr << "Could not open the file - '"
             << path << "'" << std::endl;
        exit(EXIT_FAILURE);
    }
    ss << input_file.rdbuf();
    input_file.close();
    return ss.str();
}

float* read_cluster_csv_v1(int n_clusters, int cluster_size, char* cluster_file)
{
    std::string filename(cluster_file);
    std::string file_contents;
    char delimiter = ',';

    file_contents = readFileIntoString_v1(filename);

    std::istringstream sstream(file_contents);
    float* items = (float*) malloc(n_clusters * cluster_size * sizeof(float));
    std::string record;

    int i = 0;
    while (std::getline(sstream, record)) {
        std::istringstream line(record);
        while (std::getline(line, record, delimiter)) {
            items[i] = ::atof(record.c_str());
            i++;
        }
    }

    return items;
}

__global__ void random_color_v1(uchar* colors, size_t size) {
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x >= size)
        return;
    
    curandStatePhilox4_32_10_t state;
    curand_init(42, /* the seed controls the sequence of random values that are produced */
                x, /* the sequence number is only important with multiple cores */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &state);
    colors[x] = curand(&state) % 256;
}

__global__ void change_pixels_v1(uchar* img, size_t pitch, size_t width, size_t height, uchar* colors, uchar* neighbor) {
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;
    
    uchar* pix = &img[y * pitch + x * 3];

    size_t nb_tiles_x = width / 16;
    size_t tile_x = x / 16;
    size_t tile_y = y / 16;
    size_t hist_index = tile_y * nb_tiles_x + tile_x;
    uchar cluster_index = neighbor[hist_index];
    uchar* color = colors + 3 * cluster_index;

    for (size_t i = 0; i < 3; i++)
        pix[i] = color[i];
}

int step_2_v1(uchar* cpu_img, size_t width, size_t height, short* hists, size_t h_pitch, uchar* gpu_img, size_t i_pitch, char* cluster_file) {
    size_t n_clusters = 16;
    size_t cluster_size = 256;

    size_t nb_tiles_x = width / 16;
    size_t nb_tiles_y = height / 16;
    size_t hists_size = nb_tiles_x * nb_tiles_y;

    // Lecture du fichier des clusters et stockage RAM
    float* clusters_cpu = read_cluster_csv_v1(n_clusters, cluster_size, cluster_file);
    // Copie des données clusters sur device (VRAM)
    float* clusters;
    size_t c_pitch;

    checkErr(cudaMallocPitch(&clusters, &c_pitch, cluster_size * sizeof(float), n_clusters));

    checkErr(cudaMemcpy2D(clusters, c_pitch,
                            clusters_cpu, cluster_size * sizeof(float),
                            cluster_size * sizeof(float), n_clusters,
                            cudaMemcpyHostToDevice));
    
    // Get Neighbors
    
    size_t bsize = 32;
    size_t w     = std::ceil((float)hists_size / bsize);

    // On renvoie comment la data ? On fait un nouveau tableau ou pas ? On stocke dans l'image ? Quel channel ?
    // Version simple
    uchar* neighbor;
    checkErr(cudaMalloc(&neighbor, hists_size * sizeof(uchar)));

    get_neighbor_v1<<<w, bsize>>>(clusters, c_pitch, n_clusters, cluster_size, hists, h_pitch, hists_size, neighbor);

    checkKernel();
    cudaDeviceSynchronize();
    
    // On attribue une couleur à chaque cluster
    uchar* colors;
    checkErr(cudaMalloc(&colors, 3 * sizeof(uchar) * n_clusters));

    w = std::ceil((float)3 * n_clusters / bsize);
    random_color_v1<<<w, bsize>>>(colors, 3 * n_clusters);

    checkKernel();
    cudaDeviceSynchronize();
    
    w     = std::ceil((float)width / bsize);
    size_t h     = std::ceil((float)height / bsize);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    change_pixels_v1<<<dimGrid, dimBlock>>>(gpu_img, i_pitch, width, height, colors, neighbor);

    checkKernel();
    cudaDeviceSynchronize();

    // On remets l'image en CPU
    checkErr(cudaMemcpy2D(cpu_img, 3 * width * sizeof(uchar),
                            gpu_img, i_pitch,
                            3 * width * sizeof(uchar), height,
                            cudaMemcpyDeviceToHost));

    checkErr(cudaFree(hists));
    checkErr(cudaFree(clusters));
    checkErr(cudaFree(gpu_img));
    checkErr(cudaFree(colors));
    checkErr(cudaFree(neighbor));

    return 0;
}