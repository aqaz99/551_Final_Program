// Example code taken from https://www.youtube.com/watch?v=2EbHSCvGFM0

#include <stdio.h>
// #include <math.h>
#define SIZE	655360000
#define THREADS 1024

// __global__ Lets compiler know we can run this function on GPU.
__global__ void VectorAdd(int *a, int *b, int *c, unsigned int n)
{
	// Split up work
	unsigned int work_start = (threadIdx.x) * (SIZE/THREADS);
	unsigned int work_stop = (threadIdx.x + 1) * (SIZE/THREADS);

	// Don't need for loop anymore
	// for (i=0; i < n; ++i)
	for(int i = work_start; i < work_stop; i++)
		c[i] = a[i] + b[i];
}

int main()
{
	int *a, *b, *c;
	
	cudaMallocManaged(&a, SIZE * sizeof(unsigned int));
	cudaMallocManaged(&b, SIZE * sizeof(unsigned int));
	cudaMallocManaged(&c, SIZE * sizeof(unsigned int));
	
	for (int i = 0; i < SIZE; ++i)
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}
	
	// Run the function VectorAdd with CUDA, using 1 thread block and SIZE(1024) threads per block
	VectorAdd <<<2, THREADS>>>(a, b, c, SIZE);

	// Like join? Or barrier?
	cudaDeviceSynchronize();

	// for (int i = 0; i < SIZE; ++i)
	// 	printf("c[%d] = %d\n", i, c[i]);

	// printf("c[%d] = %d\n", 1, c[1]); 
	printf("c[%d] = %d\n", SIZE-1, c[SIZE-1]);
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	return 0;
}