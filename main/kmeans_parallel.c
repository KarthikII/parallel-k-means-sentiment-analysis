#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>


#define MAX_LINE_LENGTH 1024
#define MAX_VECTOR_SIZE 37   //Number of features in a vector
#define MAX_NUM_VECTORS 100 //Number of vectors in dataset

#define MAX_SAMPLES 100 //Number of vectors taken for computation
#define FEATURES 37   // Number of features in a vector
#define K 3          // Number of clusters (Positive, Neutral, Negative)



//Data structure
typedef struct {
    float features[FEATURES];
    int cluster;  
} DataPoint;

//Pre-processing
// Function to parse a vector string and store it in a 2D array
int parse_vector(const char *vector_str, float vector[MAX_VECTOR_SIZE]) {
    int index = 0;
    char *token;
    char vector_copy[MAX_LINE_LENGTH];

    strncpy(vector_copy, vector_str, MAX_LINE_LENGTH);

    if (vector_copy[0] == '[') memmove(vector_copy, vector_copy + 1, strlen(vector_copy));
    if (vector_copy[strlen(vector_copy) - 1] == ']') vector_copy[strlen(vector_copy) - 1] = '\0';

    
    token = strtok(vector_copy, ",");
    while (token != NULL && index < MAX_VECTOR_SIZE) {
        vector[index++] = atof(token);  
        token = strtok(NULL, ",");
    }

    return index;  
}

// Function to calculate euclidean distance between two points
float distance(DataPoint a, DataPoint b) {
    float sum = 0.0;
    for (int i = 0; i < FEATURES; i++) {
        sum += (a.features[i] - b.features[i]) * (a.features[i] - b.features[i]);
    }
    return sqrt(sum);
}

// Function to update centroids based on the data points assigned to each cluster
void updateCentroids(DataPoint points[], DataPoint centroids[], int n) {
    int count[K] = {0};
    DataPoint newCentroids[K] = {};

    #pragma omp parallel for reduction(+:count[:K])
    // Sum up points assigned to each cluster
    for (int i = 0; i < n; i++) {
        int cluster = points[i].cluster;
        for (int j = 0; j < FEATURES; j++) {
            newCentroids[cluster].features[j] += points[i].features[j];
        }
        count[cluster]++;
    }

    #pragma omp parallel for
    // Update centroids
    for (int i = 0; i < K; i++) {
        if (count[i] != 0) {
            for (int j = 0; j < FEATURES; j++) {
                newCentroids[i].features[j] /= count[i];
            }
        }
    }

    // Copy the new centroids back to the original centroids array
    for (int i = 0; i < K; i++) {
        centroids[i] = newCentroids[i];
    }
}

// Function to assign each data point to the nearest centroid
void assignClusters(DataPoint points[], DataPoint centroids[], int n) {
    #pragma omp parallel for 
    for (int i = 0; i < n; i++) {
        float minDist = distance(points[i], centroids[0]);
        int minIndex = 0;
        for (int j = 1; j < K; j++) {
            float dist = distance(points[i], centroids[j]);
            if (dist < minDist) {
                minDist = dist;
                minIndex = j;
            }
        }
        points[i].cluster = minIndex;
    }
}

// K-Means clustering function
void kMeans(DataPoint points[], DataPoint centroids[], int n) {
    int iterations=10;
    #pragma omp parallel for
    for(int x=0;x<iterations;x++) {
        assignClusters(points, centroids, n);
        updateCentroids(points, centroids, n);
    }
}

int main(){
    clock_t start,end;
    
    start=clock();

    FILE *file;
    char line[MAX_LINE_LENGTH];
    float vectors[MAX_NUM_VECTORS][MAX_VECTOR_SIZE];
    int vector_lengths[MAX_NUM_VECTORS];  
    int num_vectors = 0;

    file = fopen("vector.csv", "r");
    if (file == NULL) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
        perror("Error reading header");
        fclose(file);
        return EXIT_FAILURE;
    }

    while (fgets(line, MAX_LINE_LENGTH, file) != NULL && num_vectors < MAX_NUM_VECTORS) {
        line[strcspn(line, "\n")] = 0;

        vector_lengths[num_vectors] = parse_vector(line, vectors[num_vectors]);
        num_vectors++;
    }

    fclose(file);

    float temp={};
    DataPoint points[MAX_SAMPLES];

    for(int i = 0;i < MAX_NUM_VECTORS;i++){
        for(int j=0;j<FEATURES;j++){

            points[i].features[j] = vectors[i][j];
        }

        points[i].cluster = -1;
    }

    // Initialize centroids 
    DataPoint centroids[K];

    //Taking vector 2 as positive cluster (Cluster 0)
    for(int j = 0;j < FEATURES;j++){
        centroids[0].features[j] = vectors[2][j];
    }
    centroids[0].cluster=0;

    //Taking vector 0 as negative cluster (Cluster 1)
    for(int j = 0;j < FEATURES;j++){
        centroids[1].features[j] = vectors[0][j];
    }
    centroids[1].cluster=1;

    //Taking vector 1 as neutral cluster (Cluster 2)
    for(int j = 0;j < FEATURES;j++){
        centroids[2].features[j] = vectors[1][j];
    }
    centroids[2].cluster=2;
    //printf("a");
    // Perform K-Means clustering
    kMeans(points, centroids, MAX_SAMPLES);
    end=clock();
    double time;
    time= ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\nTime: %f",time);
    printf("\n%d",points[0].cluster);
    return 0;
}

