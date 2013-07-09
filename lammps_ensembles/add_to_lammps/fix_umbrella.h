/* ----------------------------------------------------------------------
   Umbrella Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(umbrella,FixUmbrella)

#else

#ifndef FIX_UMBRELLA_H
#define FIX_UMBRELLA_H

#include "fix.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

#define GTP_COORD  0 
#define GTP_CEC    1
#define GTP_ATOM   2
#define GTP_CM     3
#define GTP_CG     4
#define GTP_CECV2  5

#define COORD_CART      0
#define COORD_SPHERICAL 1
#define COORD_CYLINDER  2
#define COORD_PT        3

class FixUmbrella : public Fix 
{
 public:
  FixUmbrella(class LAMMPS *, int, char **);
  virtual ~FixUmbrella();
  
  int me;
  int setmask();
  void init();
  void setup(int);
  void pre_force(int);
  
  /* Global setting */
  int cec;                 // Is having CEC
  int type_grp[2];         // Type of group A & B
  
  /* For atom */
  int atom_id[2];          // Atom index
  int atom_rank[2];        // Atom location
  int atom_tag[2];         // Atom global tag
  
  /* For CEC */
  int cplx_id[2];                 // ID of EVB_complex
  class EVB_Complex* cplx[2];     // Pointer of EVB_complex
  class FixEVB *fix_evb;          // Pointer of EVB module
  
  /* For CM or CG */
  int natms[2];
  int *tag_list[2];
  int *id_list[2];
  double *factor_list[2];
  double **r[2], **_r[2];
  
  /* Potential setting */
  int coord;               // Umbrella-potential coordinate
  int di[3];               // Are potential on x, y, z - direction
  double k[3];             // force constants
  double ref[3];           // Reference distance
  
  double diff,x[2][3],f[2][3],dx[3],dx2[3];
  double energy, ff[3];
  
  /* output */
  int next_out,freq_out;

  // ** AWGL: for REUS ** //
  double bias_energy;
  void compute_bias_stuff_for_external(double*, double*, double*);
  double compute_scalar();
  double compute_array(int, int);
  void modify_fix(int, double*, char*);
  double h_save;
  
 private:
  FILE *InFile, *OutFile;
  void data_umbrella();
  void compute();
  void write_log();
  
  void init_group(int);
  void setup_group(int);
  void cal_grp_force(int);
};

/*------------------------------------------------------------------------
  
  Non-spherical: U = SUM(i=x,y,z) {1/2 Ki [ r(i,A) - r(i,B) - diff(i) ]^2}

  Spherical: U = 1/2 K { SQRT( SUM(i=x,y,z){ [r(i,A)-r(i,B)]^2 } ) -diff }^2 

  ------------------------------------------------------------------------*/

}

#endif
#endif
