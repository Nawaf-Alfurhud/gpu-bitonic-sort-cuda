#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h> 
#include <time.h> 

#define N 16   // N must be a power of 2 (2, 4, 8, 16, ...)
// this is where you change the size of the array, the numbers are chosen randomly

// Device kernel: perform one bitonic step (one stage)
// Parameters:
// d_data : pointer to the array in the device memory
// j : distance between partners (stride)
// k : current bitonic sequence size
// n : total number of elements
__global__ void bitonicStep(int* d_data, int j, int k, int n)
{
    // Global index of this thread (which element in the array it handles)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;   // ignore threads outside the array

    // For this stage
    // seq = length of the small local pattern (inside a k-sized bitonic subsequence)
    // that this stage uses for its compare swap pairs.
    int seq = 2 * j;

    // Position of i inside its seq (0 .. seq-1). We get it with modulo.
    int posInSeq = i % seq;

    // Only the first half of each seq is active in this stage.
    // If posInSeq < j, then this index should do the compare swap.
    if (posInSeq < j) {

        // The partner is j positions ahead inside the same seq.
        int partner = i + j;
        if (partner >= n) return; 

        // Decide if ascending or descending for this pair
        int Segment = i / k;                 // which k-sized segment this index is in
        bool ascending = (Segment % 2 == 0); // even segments ascend, odd segments descend

        int LeftValue = d_data[i];
        int RightValue = d_data[partner];

        if (ascending) {
            // ascending: smaller value should stay at the lower index
            if (LeftValue > RightValue) {
                d_data[i] = RightValue;
                d_data[partner] = LeftValue;
            }
        }
        else {
            // descending: larger value should stay at the lower index
            if (LeftValue < RightValue) {
                d_data[i] = RightValue;
                d_data[partner] = LeftValue;
            }
        }
    }
}

int main()
{
    // h_data is the unsorted array, coming from the host
    int h_data[N];

    // Initialize the random number generator with the current time,
    // so that we get different random values every time we run the program
    srand((unsigned int)time(NULL));
    //filling the array with random numbers
    for (int i = 0; i < N; ++i) {
        h_data[i] = rand() % 10000;  
    }

    const int Size = N * sizeof(int);// size of the array, needed to allocate the memory.

    printf("Input:  ");// just printing the input for clarity.
    for (int i = 0; i < N; ++i) printf("%d ", h_data[i]);
    printf("\n\n");
    
    int* d_data = nullptr; // will point to the memory in the device(GPU)
    cudaMalloc(&d_data, Size); // allocate array on device
    cudaMemcpy(d_data, h_data, Size, cudaMemcpyHostToDevice); // copy input from host to device

    // Configure grid. I choose 32 threads per block for simplicity.
    // blocks is computed so that we have at least N threads in total.
    int threadsPerBlock = 32;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    // Outer loop (STEP): k = size of subsequences (2, 4, 8, ..., N)
    for (int k = 2; k <= N; k= k * 2) {
        // Inner loop (STAGE): j = distance between compared elements (k/2, k/4, ..., 1)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h> 
#include <time.h> 

#define N 16   // N must be a power of 2 (2, 4, 8, 16, ...)
// this is where you change the size of the array, the numbers are chosen randomly

// Device kernel: perform one bitonic step (one stage)
// Parameters:
// d_data : pointer to the array in the device memory
// j : distance between partners (stride)
// k : current bitonic sequence size
// n : total number of elements
        __global__ void bitonicStep(int* d_data, int j, int k, int n)
        {
            // Global index of this thread (which element in the array it handles)
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;   // ignore threads outside the array

            // For this stage
            // seq = length of the small local pattern (inside a k-sized bitonic subsequence)
            // that this stage uses for its compare swap pairs.
            int seq = 2 * j;

            // Position of i inside its seq (0 .. seq-1). We get it with modulo.
            int posInSeq = i % seq;

            // Only the first half of each seq is active in this stage.
            // If posInSeq < j, then this index should do the compare swap.
            if (posInSeq < j) {

                // The partner is j positions ahead inside the same seq.
                int partner = i + j;
                if (partner >= n) return;

                // Decide if ascending or descending for this pair
                int Segment = i / k;                 // which k-sized segment this index is in
                bool ascending = (Segment % 2 == 0); // even segments ascend, odd segments descend

                int LeftValue = d_data[i];
                int RightValue = d_data[partner];

                if (ascending) {
                    // ascending: smaller value should stay at the lower index
                    if (LeftValue > RightValue) {
                        d_data[i] = RightValue;
                        d_data[partner] = LeftValue;
                    }
                }
                else {
                    // descending: larger value should stay at the lower index
                    if (LeftValue < RightValue) {
                        d_data[i] = RightValue;
                        d_data[partner] = LeftValue;
                    }
                }
            }
        }

        int main()
        {
            // h_data is the unsorted array, coming from the host
            int h_data[N];

            // Initialize the random number generator with the current time,
            // so that we get different random values every time we run the program
            srand((unsigned int)time(NULL));
            //filling the array with random numbers
            for (int i = 0; i < N; ++i) {
                h_data[i] = rand() % 10000;
            }

            const int Size = N * sizeof(int);// size of the array, needed to allocate the memory.

            printf("Input:  ");// just printing the input for clarity.
            for (int i = 0; i < N; ++i) printf("%d ", h_data[i]);
            printf("\n\n");

            int* d_data = nullptr; // will point to the memory in the device(GPU)
            cudaMalloc(&d_data, Size); // allocate array on device
            cudaMemcpy(d_data, h_data, Size, cudaMemcpyHostToDevice); // copy input from host to device

            // Configure grid. I choose 32 threads per block for simplicity.
            // blocks is computed so that we have at least N threads in total.
            int threadsPerBlock = 32;
            int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
            // Outer loop (STEP): k = size of subsequences (2, 4, 8, ..., N)
            for (int k = 2; k <= N; k = k * 2) {
                // Inner loop (STAGE): j = distance between compared elements (k/2, k/4, ..., 1)

                for (int j = k; j > 0; j = j / 2) {
                    bitonicStep << <blocks, threadsPerBlock >> > (d_data, j, k, N);
                    cudaDeviceSynchronize(); // wait: all swaps of this stage must finish
                    // the following is purely to show the process of bitonic for each step and stage
                    // this degrades the perfomance a lot but it is here for the goal of
                    // showing the underlying changes of the bitonic sort algorithm
                    // these host copies and prints can easily be removed without
                    // affecting the correctness of the bitonic sort algorithm
                    cudaMemcpy(h_data, d_data, Size, cudaMemcpyDeviceToHost);// copy the data from the device to the host
                    printf("k = %d, j = %d: ", k, j);
                    for (int i = 0; i < N; ++i) printf("%d ", h_data[i]);
                    printf("\n\n");
                }
            }

            // may be redundent from the above loop but i left it here
            // is a must if you delete the above cudaMemcpy
            cudaMemcpy(h_data, d_data, Size, cudaMemcpyDeviceToHost);

            printf("Output: ");
            for (int i = 0; i < N; ++i) printf("%d ", h_data[i]);
            printf("\n");

            cudaFree(d_data);
            return 0;
        }
        {
            bitonicStep<<<blocks, threadsPerBlock >>>(d_data, j, k, N);
            cudaDeviceSynchronize(); // wait: all swaps of this stage must finish
            // the following is purely to show the process of bitonic for each step and stage
            // this degrades the perfomance a lot but it is here for the goal of
            // showing the underlying changes of the bitonic sort algorithm
            // these host copies and prints can easily be removed without
            // affecting the correctness of the bitonic sort algorithm
            cudaMemcpy(h_data, d_data, Size, cudaMemcpyDeviceToHost);// copy the data from the device to the host
            printf("k = %d, j = %d: ", k, j);
            for (int i = 0; i < N; ++i) printf("%d ", h_data[i]);
            printf("\n\n");
        }
    }

    // may be redundent from the above loop but i left it here
    // is a must if you delete the above cudaMemcpy
    cudaMemcpy(h_data, d_data, Size, cudaMemcpyDeviceToHost);

    printf("Output: ");
    for (int i = 0; i < N; ++i) printf("%d ", h_data[i]);
    printf("\n");

    cudaFree(d_data);
    return 0;
}
