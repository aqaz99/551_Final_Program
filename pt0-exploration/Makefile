INCLUDE_DIRS =
LIB_DIRS = 
CC=gcc
#CC=icc

CDEFS= 
CFLAGS= -g -Wall $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= stationary_test.c 
CUFILES= hello_cuda.cu

SRCS= ${HFILES} ${CFILES} ${CUFILES}
OBJS= ${CFILES:.c=.o}

all:	stationary_test hello_cuda hello_unthreaded

clean:
	-rm -f *.o *.d
	-rm -f stationary_test hello_cuda hello_unthreaded

stationary_test: stationary_test.o
	gcc stationary_test.c -lm -o stationary_test

hello_cuda: hello_cuda.cu
	nvcc hello_cuda.cu -o hello_cuda

hello_unthreaded: hello_unthreaded.o
	gcc hello_unthreaded.c -lm -o hello_unthreaded

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
