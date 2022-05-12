#include <stdio.h>
#include <stdlib.h>
#define SIZE	655360000

void VectorAdd(int *a, int *b, int *c, unsigned int n)
{
	int i;

	for (i=0; i < n; ++i)
		c[i] = a[i] + b[i];
}

int main()
{
	int *a, *b, *c;
	
	a = (int *)malloc(SIZE * sizeof(unsigned int));
	b = (int *)malloc(SIZE * sizeof(unsigned int));
	c = (int *)malloc(SIZE * sizeof(unsigned int));
	
	for (int i = 0; i < SIZE; ++i)
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}
	
	VectorAdd(a, b, c, SIZE);

	// for (int i = 0; i < 10; ++i)
	// 	printf("c[%d] = %d\n", i, c[i]);

    printf("c[%d] = %d\n", SIZE-1, c[SIZE-1]);
	free(a);
	free(b);
	free(c);

	return 0;
}