#include <stdlib.h>
#include <iostream>
 
#define LEN_CLUSTERS 10
#define LEN_VECT 3

// calculate the Euclidean distance between two vectors
float euclidean_distance(float *vect1, float *vect2, int len_vect){
	float distance = 0.0;
	for (int i = 0; i < len_vect; i++) 
		distance += (vect1[i] - vect2[i]) * (vect1[i] - vect2[i]);
	return distance;
}

// Locate the most similar neighbors
int get_neighbor(float clusters[LEN_CLUSTERS][LEN_VECT], float *test, int len_clusters, int len_vect){
	float *distances = (float *) malloc(len_clusters * sizeof(float));
	for (int i = 0; i < len_clusters; i++) {
		float dist = euclidean_distance(clusters[i], test, len_vect);
        distances[i] = dist;
    }
    int mini = 0;
    for (int i = 0; i < len_clusters; i++)
        if (distances[i] < distances[mini] )
            mini = i;
    free(distances);
    return mini;
}
 
 // Test distance function
int main() {
    float clusters[LEN_CLUSTERS][LEN_VECT] = {{2.7810836,2.550537003,0},
        {1.465489372,2.362125076,0},
        {3.396561688,4.400293529,0},
        {1.38807019,1.850220317,0},
        {3.06407232,3.005305973,0},
        {7.627531214,2.759262235,1},
        {5.332441248,2.088626775,1},
        {6.922596716,1.77106367,1},
        {8.675418651,-0.242068655,1},
        {7.673756466,3.508563011,1}};

    float *test = clusters[0];
    int neighbor = get_neighbor(clusters, test, LEN_CLUSTERS, LEN_VECT);
    return 0;
}