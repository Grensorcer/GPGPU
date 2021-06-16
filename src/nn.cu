#include <stdlib.h>
#include <iostream>

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
        distances[i] = dist;
    }
    int mini = 0;
    for (int i = 0; i < len_clusters; i++)
        if (distances[i] < distances[mini] )
            mini = i;

    checkErr(cudaFree(distance));
    // return mini;
}

void step_2() {
    // Lecture du fichier des clusters et stockage RAM
    // Copie des données clusters sur device (VRAM)
    // (On suppose que les informations des patches sont toujours chargées dans la device)
    // Get Neighbors

    // On renvoie comment la data ? On fait un nouveau tableau ou pas ? On stocke dans l'image ? Quel channel ?

    // On attribue une couleur à chaque cluster
    // On recontruit l'image
}