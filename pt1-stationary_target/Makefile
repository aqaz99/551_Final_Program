INCLUDE_DIRS =
LIB_DIRS = 
CC=gcc
#CC=icc

CDEFS= 
CFLAGS= -g -Wall $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= stationary_seq.c 
CUFILES= stationary_cuda.cu

SRCS= ${HFILES} ${CFILES} ${CUFILES}
OBJS= ${CFILES:.c=.o}

all:	stationary_seq stationary_cuda

clean:
	-rm -f *.o *.d
	-rm -f stationary_seq stationary_cuda

stationary_seq: stationary_seq.o
	gcc stationary_seq.c -o stationary_seq -lm

stationary_cuda: stationary_cuda.cu
	nvcc stationary_cuda.cu -o stationary_cuda -lm

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
