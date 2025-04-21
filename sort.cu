
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <limits>
#include <stdio.h>
#include <chrono>


__global__ void sort(int* dev_d, int* dev_c, int col, int row) 
{
    int** con = new int* [row];
    for (int i = 0; i < row; i++) {
        con[i] = new int[col];
    }

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            con[i][j] = dev_c[i * col + j];
        }
    }
    
    int id = threadIdx.x;
    bool swapped;

    for (int i = 0; i < col - 1; i++) {
        swapped = false;
        for (int j = 0; j < col - i - 1; j++) {
            if (con[id][j] > con[id][j + 1]) {
                int temp = con[id][j];
                con[id][j] = con[id][j + 1];
                con[id][j + 1] = temp;
                swapped = true;
            }
        }
        if (!swapped) {
            break;
        }
    }
    
        for (int j = 0; j < col; j++) {
            dev_d[id * col + j] = con[id][j];
        }
}

cudaError_t Parallel_Sort(std::vector<std::vector<int>> * c, int* d)
{
    int row = (*c).size();
    int col = 0;
    //make new_c
    //-------------------------------------------------------
    for (int i = 0; i < (*c).size(); i++) {
        if ((*c)[i].size() > col) {
            col = (*c)[i].size();
        }
    }
    int** new_c = new int* [row];
    for (int i = 0; i < row; i++) {
        new_c[i] = new int[col];
    }
    //-------------------------------------------------------
    //set new c
    //-------------------------------------------------------
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++) {
            if (j >= (*c)[i].size()) {
                new_c[i][j] = INT_MAX;
            }
            else {
                new_c[i][j] = (*c)[i][j];
            }
        }
    }
    //-------------------------------------------------------
    // Flattening Process
    //-------------------------------------------------------
    int * flattened_c = new int[row * col];
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            flattened_c[i * col + j] = new_c[i][j];
        }
    }


    int* dev_c = 0;
    int* dev_d = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);

    cudaStatus = cudaMalloc((void**)&dev_c, row * col * sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_d, row * col * sizeof(int));

    cudaStatus = cudaMemcpy(dev_c, flattened_c, row * col * sizeof(int), cudaMemcpyHostToDevice);
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "2 failed!\n");
    }
    sort << <1, row >> > (dev_d, dev_c, col, row);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "3 failed!\n");
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "4 failed!\n");
    }

    cudaStatus = cudaMemcpy(d, dev_d, col * row * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "5 failed!\n");
    }
    for (int i = 0; i < (*c).size(); i++) {
        for (int j = 0; j < (*c)[i].size(); j++) {
            (*c)[i][j] = d[col*i + j];
        }
    }


    return cudaStatus;

     
}


int main()
{
    
    const std::vector<int> a = { 1, 2, 3, 4, 5, -5, 10, 20, -42, 78, -1203};
    const std::vector<int> b = { 10, 2, 30, 40, 50, 3, 0 };
    const std::vector<int> e = { 213,213,23,13,5,45,765,3,5,756,3,655,42,1,9,1 };
    std::vector<std::vector<int>> c = {a,b, e};


    int * d = new int[a.size()*b.size()];
    auto start = std::chrono::high_resolution_clock::now();

    cudaError_t cudaStatus = Parallel_Sort(&c, d);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
  

    for (int i = 0; i < c.size(); i++) {
        printf("{");
        for (int j = 0; j < c[i].size(); j++) {
            printf("%d, ", c[i][j]);
        }
        printf("}\n");

    }
    auto end = std::chrono::high_resolution_clock::now();

    printf("%d miliseconds", std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }


    return 0;
}



