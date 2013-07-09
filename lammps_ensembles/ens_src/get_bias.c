#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "library.h"
#include "replica.h"

double get_bias(void *lmp, char* fix, int flag_skip) 
{
  if (flag_skip) return 0.0; // optional skip
  else if ( strcmp(fix, "none") == 0 ) return 0.0; // also skip if fix is none

  double *bias_ptr = (double *)lammps_extract_fix(lmp, fix, 0, 0, 0, 0);
  double bias = *bias_ptr;
  if (bias_ptr == NULL) {
    printf("Error in finding colvar fix id: %s\n", fix);
    exit(1);
  }
  bias = *bias_ptr;
  free(bias_ptr); // need to free b/c LAMMPS allocates
  return bias;
}
