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
__device__ float euclidean_distance(float *vect1, rtype vect2, int len_vect){
	float distance = 0.0;
	for (size_t i = 0; i < len_vect; i++) 
		distance += (vect1[i] - vect2[i]) * (vect1[i] - vect2[i]);
	return distance;
}

// Locate the most similar neighbors
__global__ void get_neighbor(float *clusters, size_t cluster_pitch, size_t len_clusters, size_t len_vect,
                            rtype patches, size_t patches_pitch, size_t n_patches, uint8_t *neighbor) {
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;

    if (x >= n_patches)
        return;

    float distance_mini = -1;
    uint8_t mini = 0;

    rtype patch = (rtype) ((char*) patches + x * patches_pitch);

	for (size_t i = 0; i < len_clusters; i++) {
        float* cluster = (float*) ((char*) clusters + i * cluster_pitch);

		float dist = euclidean_distance(cluster, patch, len_vect);
        if (dist < distance_mini || distance_mini == -1) {
            distance_mini = dist;
            mini = i;
        }
    }
    neighbor[x] = mini;
}

std::string readFileIntoString(const std::string& path) {
    auto ss = std::ostringstream();
    std::ifstream input_file(path);
    if (!input_file.is_open()) {
        std::cerr << "Could not open the file - '"
             << path << "'" << std::endl;
        exit(EXIT_FAILURE);
    }
    ss << input_file.rdbuf();
    return ss.str();
}

float* read_cluster_csv(int n_clusters, int cluster_size, char* cluster_file)
{
    std::string filename(cluster_file);
    std::string file_contents;
    char delimiter = ',';

    file_contents = readFileIntoString(filename);

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


rtype read_hist_csv()
{
    std::string filename("release/out.csv");
    std::string file_contents;
    char delimiter = ',';

    file_contents = readFileIntoString(filename);

    std::istringstream sstream(file_contents);
    std::vector<dtype> items;
    std::string record;

    while (std::getline(sstream, record)) {
        std::istringstream line(record);
        while (std::getline(line, record, delimiter)) {
            if (record == "val")
                continue;
            items.push_back(std::stoi(record));
        }
    }
    rtype res = (rtype) malloc(items.size() * sizeof(dtype));
    for (size_t i = 0; i < items.size(); i++)
        res[i] = items[i];
    return res;
}

__global__ void random_color(uint8_t* colors, size_t size) {
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

__global__ void change_pixels(uint8_t* img, size_t pitch, size_t width, size_t height, uint8_t* colors, uint8_t* neighbor) {
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    size_t y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height)
        return;
    
    uint8_t* pix = &img[y * pitch + x * 3];

    size_t nb_tiles_x = width / 16;
    size_t tile_x = x / 16;
    size_t tile_y = y / 16;
    size_t hist_index = tile_y * nb_tiles_x + tile_x;
    uint8_t cluster_index = neighbor[hist_index];
    uint8_t* color = colors + 3 * cluster_index;

    for (size_t i = 0; i < 3; i++)
        pix[i] = color[i];
}

int step_2(rtype hists_cpu, uint8_t* cpu_img, size_t width, size_t height, char* cluster_file) {
    size_t n_clusters = 16;
    size_t cluster_size = 256;

    size_t nb_tiles_x = width / 16;
    size_t nb_tiles_y = height / 16;
    size_t hists_size = nb_tiles_x * nb_tiles_y;

    // Lecture du fichier des clusters et stockage RAM
    float* clusters_cpu = read_cluster_csv(n_clusters, cluster_size, cluster_file);
    // Copie des donn??es clusters sur device (VRAM)
    float* clusters;
    size_t c_pitch;

    checkErr(cudaMallocPitch(&clusters, &c_pitch, cluster_size * sizeof(float), n_clusters));

    checkErr(cudaMemcpy2D(clusters, c_pitch,
                            clusters_cpu, cluster_size * sizeof(float),
                            cluster_size * sizeof(float), n_clusters,
                            cudaMemcpyHostToDevice));

    // (On suppose que les informations des patches sont toujours charg??es dans la device)
    // En fait on suppose pas on charge depuis CPU genre cplussimple
    rtype hists;
    size_t h_pitch;
    
    checkErr(cudaMallocPitch(&hists, &h_pitch, cluster_size * sizeof(dtype), hists_size));

    checkErr(cudaMemcpy2D(hists, h_pitch,
                            hists_cpu, cluster_size * sizeof(dtype),
                            cluster_size * sizeof(dtype), hists_size,
                            cudaMemcpyHostToDevice));

    
    // Get Neighbors
    
    size_t bsize = 32;
    size_t w     = std::ceil((float)hists_size / bsize);

    // On renvoie comment la data ? On fait un nouveau tableau ou pas ? On stocke dans l'image ? Quel channel ?
    // Version simple
    uint8_t* neighbor;
    checkErr(cudaMalloc(&neighbor, hists_size * sizeof(uint8_t)));

    get_neighbor<<<w, bsize>>>(clusters, c_pitch, n_clusters, cluster_size, hists, h_pitch, hists_size, neighbor);

    checkKernel();
    cudaDeviceSynchronize();
    
    // On attribue une couleur ?? chaque cluster
    uint8_t* colors;
    checkErr(cudaMalloc(&colors, 3 * sizeof(uint8_t) * n_clusters));

    w = std::ceil((float)3 * n_clusters / bsize);
    random_color<<<w, bsize>>>(colors, 3 * n_clusters);

    checkKernel();
    cudaDeviceSynchronize();

    // On recontruit l'image
    uint8_t* gpu_img;
    size_t i_pitch;

    checkErr(cudaMallocPitch(&gpu_img, &i_pitch, 3 * width * sizeof(uint8_t), height));

    checkErr(cudaMemcpy2D(gpu_img, i_pitch,
                            cpu_img, 3 * width * sizeof(uint8_t),
                            3 * width * sizeof(uint8_t), height,
                            cudaMemcpyHostToDevice));

    
    w     = std::ceil((float)width / bsize);
    size_t h     = std::ceil((float)height / bsize);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    change_pixels<<<dimGrid, dimBlock>>>(gpu_img, i_pitch, width, height, colors, neighbor);

    checkKernel();
    cudaDeviceSynchronize();

    // On remets l'image en CPU
    checkErr(cudaMemcpy2D(cpu_img, 3 * width * sizeof(uint8_t),
                            gpu_img, i_pitch,
                            3 * width * sizeof(uint8_t), height,
                            cudaMemcpyDeviceToHost));

    checkErr(cudaFree(hists));
    checkErr(cudaFree(clusters));
    checkErr(cudaFree(gpu_img));
    checkErr(cudaFree(colors));
    checkErr(cudaFree(neighbor));

    return 0;
}