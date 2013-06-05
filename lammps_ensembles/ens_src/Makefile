#----------------------------------------------------

# User-defined values

# Compiler executable
CC = mpicc

# point to LAMMPS source directory where library has been compiled
LAMMPSDIR = /Users/alange/research/lammps/lammps-9Mar13/src 

# External library paths and identifiers
#LIB = -llammps_mira -lfftw3f
#LIBDIR = /usr/lib
LIBCOLVARS = -L/Users/alange/research/lammps/lammps-9Mar13/lib/colvars/
LIBDIR = 
LIB = -llammps_mac_awgl $(LIBCOLVARS) -lcolvars 

# Debugging flags
#DEBUG = -g -DCOORDX_DEBUG 
DEBUG = -g -DCOORDX_DEBUG -DCOORDX_DEBUG_LEVEL_TWO
#DEBUG = -g

# Compiler flags, optimizations
CCFLAGS = -O3 -fopenmp $(DEBUG) -I$(LAMMPSDIR) 

#----------------------------------------------------

# Make commands
#
SRC = $(wildcard *.c)
INC = $(wildcard *.h)
OBJ = $(SRC:.c=.o)

install: $(OBJ) 
	$(CC) $(CCFLAGS) $(LIBDIR) -L$(LAMMPSDIR) \
	$(OBJ) \
	$(LIB) -lstdc++ -lm -o ens_driver

clean:
	rm -f *.o ./ens_driver 

%.o: %.c
	$(CC) $(CCFLAGS) -c $<
