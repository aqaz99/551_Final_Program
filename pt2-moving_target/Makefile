INCLUDE_DIRS =
LIB_DIRS = 
CC=gcc
#CC=icc

CDEFS= 
CFLAGS= -g -Wall $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= moving_seq.c 
CUFILES= moving_cuda.cu

SRCS= ${HFILES} ${CFILES} ${CUFILES}
OBJS= ${CFILES:.c=.o}

all:	moving_seq moving_cuda

clean:
	-rm -f *.o *.d
	-rm -f moving_seq moving_cuda

moving_seq: moving_seq.o
	gcc moving_seq.c -o moving_seq -lm

moving_cuda: moving_cuda.cu
	nvcc moving_cuda.cu -o moving_cuda -lm

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
