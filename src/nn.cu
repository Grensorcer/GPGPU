#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
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

#if defined(naive)
    typedef unsigned short* rtype;
#else
    typedef unsigned * rtype;
#endif

// calculate the Euclidean distance between two vectors
__device__ float euclidean_distance(float *vect1, rtype vect2, int len_vect){
	float distance = 0.0;
	for (int i = 0; i < len_vect; i++) 
		distance += (vect1[i] - vect2[i]) * (vect1[i] - vect2[i]);
	return distance;
}

// Locate the most similar neighbors
__global__ void get_neighbor(float *clusters, size_t cluster_pitch, size_t len_clusters, size_t len_vect,
                            rtype patches, size_t patches_pitch, size_t n_patches, int *neighbor) {
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= n_patches || x >= 11520)
        return;

    float distance_mini = -1;
    int mini = -1;
	for (int i = 0; i < len_clusters; i++) {
		float dist = euclidean_distance(clusters + i * cluster_pitch, patches + x * patches_pitch, len_vect);
        if (dist < distance_mini || distance_mini == -1) {
            distance_mini = dist;
            mini = i;
        }
    }
    neighbor[x] = mini;
    /*if (threadIdx.x==0)
        printf("2ND x = %lu ### bon cluster_pitch = %lu len_clusters = %lu len_vect = %lu patches_pitch = %lu n_patches = %lu\n",
            x, cluster_pitch, len_clusters, len_vect, patches_pitch, n_patches);*/
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

float* read_cluster_csv(int n_clusters, int cluster_size)
{
    std::string filename("cluster.csv");
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
            items[i] = ::atof(record.c_str()) * 256;
            i++;
        }
    }

    return items;
}

/*
__global__ void check_clusters(float* clusters) {
    for (int i = 0; i < 10; i++) {
        printf("%f \n", clusters[i]);
    }
}
*/

__global__ void check_neighbor(int* neighbor) {
    printf("START\n");
    for (int i = 0; i < 10; i++) {
        printf("%d \n", neighbor[i]);
    }
}

void step_2(rtype hists_cpu, int nb_tiles_x, int nb_tiles_y) {
    size_t n_clusters = 16;
    size_t cluster_size = 256;
    size_t hists_size = nb_tiles_x * nb_tiles_y;
    std::cout << nb_tiles_x<<" " << nb_tiles_y<<" " << hists_size << std::endl;

    // Lecture du fichier des clusters et stockage RAM
    float* clusters_cpu = read_cluster_csv(n_clusters, cluster_size);
    // Copie des données clusters sur device (VRAM)
    float* clusters;
    size_t c_pitch;

    checkErr(cudaMallocPitch(&clusters, &c_pitch, cluster_size * sizeof(float), n_clusters));

    checkErr(cudaMemcpy2D(clusters, c_pitch,
                            clusters_cpu, cluster_size * sizeof(float),
                            cluster_size * sizeof(float), n_clusters,
                            cudaMemcpyHostToDevice));
    
    //check_clusters<<<1, 1>>>(clusters);

    // (On suppose que les informations des patches sont toujours chargées dans la device)
    // En fait on suppose pas on charge depuis CPU genre cplussimple
    rtype hists;
    size_t h_pitch;
    
    checkErr(cudaMallocPitch(&hists, &h_pitch, cluster_size * sizeof(unsigned short), hists_size));

    checkErr(cudaMemcpy2D(hists, h_pitch,
                            hists_cpu, cluster_size * sizeof(unsigned short),
                            cluster_size * sizeof(unsigned short), hists_size,
                            cudaMemcpyHostToDevice));

    
    // Get Neighbors
    
    size_t bsize = 256;
    size_t w     = std::ceil((float)hists_size / bsize);
    std::cout << "w:" << w << std::endl;

    // On renvoie comment la data ? On fait un nouveau tableau ou pas ? On stocke dans l'image ? Quel channel ?
    // Version simple
    int *neighbor;
    checkErr(cudaMalloc(&neighbor, hists_size * sizeof(int)));

    std::cout <<"c_pitch="<< c_pitch<<" n_clusters=" << n_clusters<<" cluster_size=" << cluster_size << " h_pitch=" << h_pitch << " hists_size="<<hists_size << std::endl;
    get_neighbor<<<w, bsize>>>(clusters, c_pitch, n_clusters, cluster_size, hists, h_pitch, hists_size, neighbor);

    checkKernel();
    cudaDeviceSynchronize();

    check_neighbor<<<1, 1>>>(neighbor);
    
    // On attribue une couleur à chaque cluster
    // On recontruit l'image

    checkErr(cudaFree(hists));
    checkErr(cudaFree(clusters));
}