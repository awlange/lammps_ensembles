#----------------------------------------------------

# User-defined values

# Compiler executable
CC = mpicc

# point to LAMMPS source directory where library has been compiled (don't leave space at end!)
LAMMPSDIR = /Users/alange/research/lammps_ensembles/LE_070813/lammps-9Mar13
LAMMPSSRC = $(LAMMPSDIR)/src

# External library paths and identifiers
#LIB = -llammps_mira -lfftw3f
#LIBDIR = /usr/lib
LIBCOLVARS = $(LAMMPSDIR)/lib/colvars/
LIBDIR = 
LIB = -llammps_mac_awgl -L$(LIBCOLVARS) -lcolvars 

# Debugging flags
DEBUG = -g -DMREUS_DEBUG
#DEBUG = -g

# Compiler flags, optimizations
CCFLAGS = -O3 -fopenmp $(DEBUG) -I$(LAMMPSSRC) 

#----------------------------------------------------

# Make commands
#
SRC = $(wildcard *.c)
INC = $(wildcard *.h)
OBJ = $(SRC:.c=.o)

install: $(OBJ) 
	$(CC) $(CCFLAGS) $(LIBDIR) -L$(LAMMPSSRC) \
	$(OBJ) \
	$(LIB) -lstdc++ -lm -o ens_driver

clean:
	rm -f *.o ./ens_driver 

%.o: %.c
	$(CC) $(CCFLAGS) -c $<
