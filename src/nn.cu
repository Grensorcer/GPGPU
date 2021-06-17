#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

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
__device__ float euclidean_distance(float *vect1, float *vect2, int len_vect){
	float distance = 0.0;
	for (int i = 0; i < len_vect; i++) 
		distance += (vect1[i] - vect2[i]) * (vect1[i] - vect2[i]);
	return distance;
}

// Locate the most similar neighbors
__global__ void get_neighbor(float *clusters, int cluster_pitch, int len_clusters, int len_vect, float *patches, int patches_pitch) {
    size_t x = blockDim.x * blockIdx.x + threadIdx.x;

    float* distance;

    checkErr(cudaMalloc(&distance, len_clusters * sizeof(float)));

	for (int i = 0; i < len_clusters; i++) {
		float dist = euclidean_distance(clusters + i * cluster_pitch, patches + x * patches_pitch, len_vect);
        distance[i] = dist;
    }
    int mini = 0;
    for (int i = 0; i < len_clusters; i++)
        if (distance[i] < distance[mini] )
            mini = i;

    checkErr(cudaFree(distance));
    // return mini;
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

void step_2(rtype hists_cpu, int nb_tiles_x, int nb_tiles_y) {
    int n_clusters = 16;
    int cluster_size = 256;
    int hists_size = nb_tiles_x * nb_tiles_y;

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
    
    checkErr(cudaMallocPitch(&hists, &h_pitch, cluster_size * sizeof(hists_cpu[0]), hists_size));

    checkErr(cudaMemcpy2D(hists, h_pitch,
                            hists_cpu, cluster_size * sizeof(hists_cpu[0]),
                            cluster_size * sizeof(hists_cpu[0]), hists_size,
                            cudaMemcpyHostToDevice));

    
    // Get Neighbors
    
    int bsize = 32;
    int w     = std::ceil((float)nb_tiles_x / bsize);
    int h     = std::ceil((float)nb_tiles_y / bsize);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    get_neighbor<<<dimGrid, dimBlock>>>(float *clusters, int cluster_pitch, int len_clusters, int len_vect, float *patches, int patches_pitch)

    // On renvoie comment la data ? On fait un nouveau tableau ou pas ? On stocke dans l'image ? Quel channel ?

    // On attribue une couleur à chaque cluster
    // On recontruit l'image

    checkErr(cudaFree(hists));
    checkErr(cudaFree(clusters));
}