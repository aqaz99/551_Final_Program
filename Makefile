INCLUDE_DIRS =
LIB_DIRS = 
CC=gcc
#CC=icc

CDEFS= 
CFLAGS= -g -Wall $(INCLUDE_DIRS) $(CDEFS)
LIBS= 

HFILES= 
CFILES= sequential_shooter.c

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.c=.o}

all:	sequential_shooter

clean:
	-rm -f *.o *.d
	-rm -f sequential_shooter

sequential_shooter: sequential_shooter.o
	gcc sequential_shooter.c -lm -o sequential_shooter

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
