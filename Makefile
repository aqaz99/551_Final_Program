INCLUDE_DIRS =
LIB_DIRS = 
CC=gcc
#CC=icc

CDEFS= 
CFLAGS= -g -Wall $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= sequential_shooter.c 
CUFILES= hello_cuda.cu

SRCS= ${HFILES} ${CFILES} ${CUFILES}
OBJS= ${CFILES:.c=.o}

all:	sequential_shooter hello_cuda hello_unthreaded

clean:
	-rm -f *.o *.d
	-rm -f sequential_shooter hello_cuda hello_unthreaded

sequential_shooter: sequential_shooter.o
	gcc sequential_shooter.c -lm -o sequential_shooter

hello_cuda: hello_cuda.cu
	nvcc hello_cuda.cu -o hello_cuda

hello_unthreaded: hello_unthreaded.o
	gcc hello_unthreaded.c -lm -o hello_unthreaded

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
