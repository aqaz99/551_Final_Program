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

rotation: sequential_shooter.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o 

depend:

.c.o:
	$(CC) $(CFLAGS) -c $<
