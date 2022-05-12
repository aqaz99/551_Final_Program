// Example code taken from https://www.youtube.com/watch?v=2EbHSCvGFM0

#include <stdio.h>
// #include <math.h>
#define SIZE	200
#define THREADS 6 // This is total thread count, not threads per block
#define BLOCKS 2 // Try and stick to even numbers

// __global__ Lets compiler know we can run this function on GPU.
__global__ void VectorAdd(int *a, int *b, int *c, unsigned int n)
{	
	// Need to get work from where the last blocks last thread's work stops
	double work_per_thread = (SIZE/THREADS);
	int threads_per_block = THREADS / BLOCKS;
	printf("work per thread %f\n", work_per_thread);

	// Split up work
	unsigned int work_start = work_per_thread * (threads_per_block * blockIdx.x + threadIdx.x);
	unsigned int work_stop;

	// Last thread pick up remainder
	if(threadIdx.x == (threads_per_block-1) && blockIdx.x == BLOCKS-1){
		work_stop = SIZE;
	}else{
		work_stop = work_per_thread * (threads_per_block * blockIdx.x + threadIdx.x + 1);
	}

	

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
	
	// Run the function VectorAdd with CUDA, using 1 thread block and THREADS/BLOCK threads per block
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