// Example code taken from https://www.youtube.com/watch?v=2EbHSCvGFM0

#include <stdio.h>
// #include <math.h>
#define SIZE	20
#define THREADS 4
#define BLOCKS 2 // Try and stick to even numbers

// __global__ Lets compiler know we can run this function on GPU.
__global__ void VectorAdd(int *a, int *b, int *c, unsigned int n)
{	
	// Need to get work from where the last blocks last thread's work stops
	int work_per_thread = (SIZE/THREADS);
	printf("work per thread %d\n", work_per_thread);
	// Split up work
	unsigned int work_start = work_per_thread * (2 * blockIdx.x + threadIdx.x);
	unsigned int work_stop = work_per_thread * (2 * blockIdx.x + threadIdx.x + 1);

	

	printf("Block [%d] thread[%d] computing from %d to %d\n", blockIdx.x, threadIdx.x, work_start, work_stop-1);

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
	VectorAdd <<<BLOCKS, THREADS/BLOCKS>>>(a, b, c, SIZE);

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