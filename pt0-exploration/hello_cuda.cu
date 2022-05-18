// Example code taken from https://www.youtube.com/watch?v=2EbHSCvGFM0

#include <stdio.h>
// #include <math.h>
#define SIZE	655360000
#define THREADS 512 // This is total thread count, not threads per block
#define BLOCKS 128 // Try and stick to even numbers

// __global__ Lets compiler know we can run this function on GPU.
__global__ void VectorAdd(unsigned int *a, unsigned  int *b, unsigned  int *c, unsigned int n)
{	
	// Need to get work from where the last blocks last thread's work stops
	double work_per_thread = (SIZE/THREADS);
	int threads_per_block = THREADS / BLOCKS;
	// printf("work per thread %f\n", work_per_thread);

	// Split up work
	unsigned int work_start = work_per_thread * (threads_per_block * blockIdx.x + threadIdx.x);
	unsigned int work_stop;

	// Last thread pick up remainder
	if(threadIdx.x == (threads_per_block-1) && blockIdx.x == BLOCKS-1){
		work_stop = SIZE;
	}else{
		work_stop = work_per_thread * (threads_per_block * blockIdx.x + threadIdx.x + 1);
	}

	// printf("Block [%d] thread[%d] computing from %d to %d\n", blockIdx.x, threadIdx.x, work_start, work_stop-1);

	for(int i = work_start; i < work_stop; i++)
		c[i] = a[i] + b[i];
	
}

int main()
{
	unsigned int *a, *b, *c;
	
	cudaMallocManaged(&a, SIZE * sizeof(unsigned int));
	cudaMallocManaged(&b, SIZE * sizeof(unsigned int));
	cudaMallocManaged(&c, SIZE * sizeof(unsigned int));
	
	// Sequential code, perhaps we could time it or thread it
	for (int i = 0; i < SIZE; ++i)
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}
	
	// I should time this function

	// Run the function VectorAdd with CUDA, using 1 thread block and THREADS/BLOCK threads per block
	VectorAdd <<<BLOCKS, THREADS/BLOCKS>>>(a, b, c, SIZE);

	// printf("Before Barrier\n");
	// Like join? Or barrier?
	cudaDeviceSynchronize();

	// printf("After Barrier\n");

	// for (int i = 0; i < SIZE; ++i)
	// 	printf("c[%d] = %d\n", i, c[i]);

	// printf("c[%d] = %d\n", 1, c[1]); 
	
	printf("Final variable in array: c[%d] = %d\n", SIZE-1, c[SIZE-1]);
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	return 0;
}