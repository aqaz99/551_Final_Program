INCLUDE_DIRS =
LIB_DIRS = 
CC=gcc
#CC=icc

CDEFS= 
CFLAGS= -g -Wall $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= stationary_seq.c 
CUFILES= 

SRCS= ${HFILES} ${CFILES} ${CUFILES}
OBJS= ${CFILES:.c=.o}

all:	stationary_seq

clean:
	-rm -f *.o *.d
	-rm -f stationary_seq

stationary_seq: stationary_seq.o
	gcc stationary_seq.c -o stationary_seq -lm

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
