/* ----------------------------------------------------------------------
   Umbrella Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

#include "atom.h"
#include "update.h"
#include "domain.h"
#include "group.h"
#include "modify.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "comm.h"
#include "output.h"

#include "fix_umbrella.h"

#include "fix_evb.h"
#include "EVB_engine.h"
#include "EVB_complex.h"
#include "EVB_cec.h"
#include "EVB_cec_v2.h"
#include "EVB_source.h"

#define MAXLENGTH 20000

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

FixUmbrella::FixUmbrella(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{  
  // Set up fix parameters
  MPI_Comm_rank(world,&me);
  virial_flag = 1;
  InFile = OutFile = 0;
  
  for(int i=0; i<2; i++) 
  {
    tag_list[i] = id_list[i] = NULL;
	factor_list[i] = NULL;
  }
  
  if (me == 0) 
  {
    if (screen) fprintf(screen,"[Umbrella] Construct Umbrella module ... ");
    
    InFile = fopen(arg[3],"r");
    OutFile = fopen(arg[4],"w");
     
    if (InFile == NULL) 
    {
      char str[128];
      sprintf(str,"Cannot open Umbrella input file %s",arg[3]);
      error->one(FLERR,str);
    }
    
    if (OutFile == NULL) 
    {
      char str[128];
      sprintf(str,"Cannot open Umbrella output file %s",arg[4]);
      error->one(FLERR,str);
    }
  }  
  
  data_umbrella();

  if (me==0)
  {
    fprintf(OutFile, "# [STEP]");
    
    if(coord==COORD_CART)
    {
      fprintf(OutFile," [U]");
      if(di[0]) fprintf(OutFile," [X1-X2] [X1] [X2]");
      if(di[1]) fprintf(OutFile," [Y1-Y2] [Y1] [Y2]");
      if(di[2]) fprintf(OutFile," [Z1-Z2] [Z1] [Z2]");
    }
    else if(coord==COORD_SPHERICAL) fprintf(OutFile," [U] [R] [X1] [Y1] [Z1] [X2] [Y2] [Z2]");
      
    fprintf(OutFile,"\n\n");
    fflush(OutFile);
    if (screen) fprintf(screen,"Finished.\n");
  }
}

/* ---------------------------------------------------------------------- */

FixUmbrella::~FixUmbrella()
{
  if (me==0) 
  {
    if (screen) fprintf(screen,"[Umbrella] Destruct Umbrella module... ");
    fclose(InFile); fclose(OutFile); 
  }
  
  for(int i=0; i<2; i++) if(type_grp[i] == GTP_CM || type_grp[i] == GTP_CG)
  {
    delete [] tag_list[i]; 
    delete [] id_list[i]; 
    delete [] factor_list[i]; 
    memory->destroy(r[i]);
    memory->destroy(_r[i]);
  }
  
  if (me==0 && screen)  fprintf(screen,"Finished.\n");
}

/* ---------------------------------------------------------------------- */

int FixUmbrella::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixUmbrella::init()
{
  fix_evb = NULL;
  
  for(int i=0; i<modify->nfix; i++)
  {
    fix_evb = (FixEVB*) modify->fix[i];
		
	if(strcmp(fix_evb->style,"evb")!=0) 
	  fix_evb = NULL;	
	else break;
  }

  if(cec && !fix_evb)
    error->all(FLERR,"\n[Umbrella] Can NOT use CEC group without EVB ojbect.\n[Umbrella] (1) May not define FixEVB ojbect.\n[Umbrella] (2) May FixUmbrella defined before FixEVB\n");

  if(cec) fix_evb->Engine->bDelayEff = true;
}

/* ---------------------------------------------------------------------- */

void FixUmbrella::setup(int vflag)
{
   
    int nlocal = atom->nlocal;
    int nall = nlocal + atom->nghost;   
    for(int i=nlocal; i<nall; i++)
    atom->f[i][0] = atom->f[i][1] = atom->f[i][2] = 0.0;
      
    pre_force(vflag);
    if(me==0) write_log();
    if(comm->me==0 && screen) fprintf(screen,"[fix_umbrella] Setup...\n");
    comm->reverse_comm();
}

/* ---------------------------------------------------------------------- */

void FixUmbrella::pre_force(int vflag)
{
   compute();

    for(int i=0; i<2; i++)
    {
        if (type_grp[i]==GTP_CEC) cplx[i]->cec->decompose_force(f[i]);
	else if (type_grp[i]==GTP_CECV2) cplx[i]->cec_v2->decompose_force(f[i]);
        else if(type_grp[i]==GTP_ATOM && atom_rank[i] == me)
        { VECTOR_SELF_ADD(atom->f[atom_id[i]],f[i]); }
        else if(type_grp[i]==GTP_CM || type_grp[i]==GTP_CG) 
            cal_grp_force(i);
    }
 
    // Test

//    comm->reverse_comm();
    /*
    int nlocal = atom->nlocal;
    for(int i=0; i<nlocal+atom->nghost; i++) 
      for(int d=0; d<3; d++)
	if(fabs(atom->f[i][d])>300.0)
    {
        fprintf(screen,"%8d %lf %lf %lf\n",update->ntimestep,atom->f[i][0],atom->f[i][1],atom->f[i][2]);
    }
/*  
    int nall = nlocal + atom->nghost;
    for(int i=0; i<nall; i++)
    {
        atom->x[i][0] += 0.1;
        atom->x[i][1] += 0.1;
        atom->x[i][2] += 0.1;
    }
*/    
    if(me==0)
    {
        if(freq_out)
        {
            if(next_out==freq_out) next_out = 1;
            else { next_out++; return; }
        }
        else if( update->ntimestep != output->next ) return;

        write_log();
    }
}

/* ---------------------------------------------------------------------- */

void FixUmbrella::data_umbrella()
{ 
  double ParaBuf[MAXLENGTH];
  int ParaCount=0;
  
  // Read in the parameters from rank=0

  if(me==0)
  {
    char line[MAXLENGTH+1];
    while(fgets(line,MAXLENGTH,InFile))
    {
      char *p=line;
      while(*p && *p!='\n' && *p!='#')
      {
        while( *p==' ' || *p=='\t' )  p++;
        if(*p=='\n' || *p=='#') continue;
        sscanf(p,"%lf",ParaBuf+ParaCount);
        ParaCount++;
		if (ParaCount == MAXLENGTH) error->one(FLERR,"[Umbrella] Too many parameters!");
        while( *p && *p!=' ' && *p!='\t' && *p!='#' )  p++;
      }
    }
  }
  
  if(me==0 && screen) fprintf(screen,"[fix_umbrella] Read %d parameters.\n",ParaCount++);
  
  // Broadcast the parameters to the world

  MPI_Bcast(&ParaCount,1,MPI_INT,0,world);
  MPI_Bcast(ParaBuf,ParaCount,MPI_DOUBLE,0,world);
  
  // Set up the parameters
  
  int m=0;
  double *buf = ParaBuf;
  
  for(int i=0; i<2; i++)
  {
    // Group Type:
    // 0 - coordinate
    // 1 - CEC
    // 2 - atom 
    // 3 - center of mass
    // 4 - center of geometry
    
    type_grp[i] = static_cast<int> (buf[m++]);
	
    if (type_grp[i] == GTP_COORD) 
      for(int j=0; j<3; j++) x[i][j] = (buf[m++]);
    else if (type_grp[i] == GTP_CEC || type_grp[i] == GTP_CECV2) 
      cplx_id[i] = static_cast<int> (buf[m++]);
    else if (type_grp[i] == GTP_ATOM)
      atom_tag[i] = static_cast<int> (buf[m++]);
    else if (type_grp[i] == GTP_CM || type_grp[i] == GTP_CG)
      {
	natms[i] = static_cast<int> (buf[m++]);
	tag_list[i] = new int [natms[i]];
	id_list[i] = new int[natms[i]];
	factor_list[i] = new double[natms[i]];
	for(int j=0; j<natms[i]; j++) tag_list[i][j] = static_cast<int> (buf[m++]);
	init_group(i);
      }
    else error->all(FLERR,"[Umbrella] Undefined group_type!");
  }
  
  if(type_grp[0]==1 || type_grp[1]==1) cec = 1;
  else cec = 0;
  
  coord = static_cast<int>(buf[m++]);
  for(int i=0; i<3; i++) di[i] = static_cast<int>(buf[m++]);
  
  if(coord==COORD_SPHERICAL)
  {    
	k[0] = buf[m++];
	ref[0] = buf[m++];
  }
  else if(coord==COORD_CART) 
  {
    for(int i=0; i<3; i++) if (di[i]) k[i] = buf[m++];
    for(int i=0; i<3; i++) if (di[i]) ref[i] = buf[m++];
  }
  else if(coord==COORD_CYLINDER)
  {
    if(type_grp[0]!=GTP_COORD && type_grp[1]!=GTP_COORD)
      error->warning(FLERR,"[Umbrella] Must use one coord-type group when using cylinder potential.\n");
    for(int i=0; i<2; i++) if (di[i]) k[i] = buf[m++];

    double mol = 0.0;
    for(int i=0; i<3; i++)
    {
      ref[i] = buf[m++];
      mol += ref[i]*ref[i];
    }
    mol = sqrt(mol);
    for(int i=0; i<3; i++) ref[i]/=mol;
  }
  else if(coord==COORD_PT)
  {
    for(int i=0; i<2; i++) k[i] = buf[m++];
    atom_tag[0] = static_cast<int>(buf[m++]);
    ref[0] = buf[m++];
  }
  else error->all(FLERR,"[Umbrella] Unkown COORD setting.");

  freq_out = static_cast<int>(buf[m++]);
  next_out = freq_out;
}

/* ---------------------------------------------------------------------- */

void FixUmbrella::compute()
{
  energy = 0.0;
  f[0][0] = f[0][1] = f[0][2] = f[1][0] = f[1][1] = f[1][2] = 0.0;
  virial[0] = virial[1] = virial[2] = virial[3] = virial[4] = virial[5] = 0.0;

  // Get index
  for(int i=0; i<2; i++)
    if(type_grp[i] == GTP_CEC)
	{
	  cplx[i] = fix_evb->Engine->all_complex[cplx_id[i]];
	  for(int j=0; j<3; j++) if(di[j]) x[i][j] = cplx[i]->cec->r_cec[j];
	}
	
	else if(type_grp[i] == GTP_CECV2)
	{
	  cplx[i] = fix_evb->Engine->all_complex[cplx_id[i]];
	  for(int j=0; j<3; j++) if(di[j]) x[i][j] = cplx[i]->cec_v2->r_cec[j];
	}
	else if(type_grp[i]== GTP_ATOM)
	{
	  atom_id[i] = atom->map(atom_tag[i]);
	  int location = me;
	  if (atom_id[i]<0 || atom_id[i]>=atom->nlocal) location = 0;
	  else 
	  { 
	    x[i][0] = atom->x[atom_id[i]][0];
	    x[i][1] = atom->x[atom_id[i]][1];
	    x[i][2] = atom->x[atom_id[i]][2];
	  }
	  
	  atom_rank[i]=0;
	  MPI_Allreduce(&location,atom_rank+i,1,MPI_INT,MPI_SUM,world);
	  MPI_Bcast(x[i],3,MPI_DOUBLE,atom_rank[i],world);
	}
	else if(type_grp[i]== GTP_CM) setup_group(i);
  
  dx[0] = dx[1] = dx[2] = 0.0;
  ff[0] = ff[1] = ff[2] = 0.0;

  /******************************************/
  if(coord==COORD_CART) 
  {
    for(int i=0; i<3; i++) if (di[i]) dx[i] = x[0][i]-x[1][i];
    VECTOR_PBC(dx);
    for(int i=0; i<3; i++) if (di[i]) dx[i] = dx[i]-ref[i];
    VECTOR_PBC(dx);
	
    for(int i=0; i<3; i++) if (di[i])
    {
	  ff[i] = -k[i];
	  f[0][i] = - k[i] * dx[i]; f[1][i] = -f[0][i];
	  dx2[i] = dx[i]*dx[i];
	  energy += 0.5 * k[i] * dx2[i];
    }

    for(int i=0; i<3; i++) if (di[i]) dx[i] = x[0][i]-x[1][i];
    VECTOR_PBC(dx);
  }
  /******************************************/
  else if(coord==COORD_SPHERICAL)
  {
    double dr2 = 0.0;
	
	for(int i=0; i<3; i++) if(di[i]) dx[i] = x[0][i]-x[1][i];
	VECTOR_PBC(dx);
	for(int i=0; i<3; i++) if(di[i]) dr2 += dx[i]*dx[i];
	
	double dr = sqrt(dr2);
	diff = dr - ref[0];
	energy = 0.5 * k[0] * diff * diff;
        
	for(int i=0; i<3; i++) if(di[i]) 
	{ 
	  ff[i] = -k[0] * diff / dr;
	  f[0][i] = ff[i] * dx[i]; 
	  f[1][i] = -f[0][i]; 
	}
  }
  /******************************************/
  else if(coord==COORD_CYLINDER)
  { 
    int a0, a1;
    if(type_grp[0]==GTP_COORD) { a0=0; a1=1; }
    else { a1=0; a0=1; }

    // vector R(real)=R1-R2;
    for(int i=0; i<3; i++) 
      dx[i] = x[a1][i]-x[a0][i];
    VECTOR_PBC(dx);

    // h direction
    double h = 0.0;
    for(int i=0; i<3; i++) h+=dx[i]*ref[i];
    energy += 0.5*k[0]*h*h;
    // AWGL
    h_save = h; // save for extraction

    double dh[3],fh[3];
    for(int i=0; i<3; i++) 
    {
      dh[i]=ref[i]*h;
      fh[i]=-k[0]*dh[i];
      f[a1][i]+=fh[i];
      f[a0][i]-=fh[i];
      virial[i]=fh[i]*dh[i]*dh[i];
    }

    // r direction
    
    double r = 0.0;
    double dr[3],fr[3];
    for(int i=0; i<3; i++)
    {
      dr[i]=dx[i]-dh[i];
      r += dr[i]*dr[i];
    }
    energy += 0.5*k[1]*r;

    for(int i=0; i<3; i++) 
    {
      fr[i]=-k[1]*dr[i];
      f[a1][i]+=fr[i];
      f[a0][i]-=fr[i];
      virial[i]=fr[i]*dr[i]*dr[i];
    } 
  }
  /******************************************/
  else if(coord==COORD_PT)
  { 
    atom_id[0] = atom->map(atom_tag[0]);
    double xto[3];
    
    int location = me;
    if (atom_id[0]<0 || atom_id[0]>=atom->nlocal) location = 0;
    else 
    { 
      xto[0] = atom->x[atom_id[0]][0];
      xto[1] = atom->x[atom_id[0]][1];
      xto[2] = atom->x[atom_id[0]][2];
    }
  
    atom_rank[0]=0;
    MPI_Allreduce(&location,atom_rank,1,MPI_INT,MPI_SUM,world);
    MPI_Bcast(xto,3,MPI_DOUBLE,atom_rank[0],world);
          
    double **ax = atom->x;  
    
    double direction[3];
    direction[0] = xto[0] - x[1][0];
    direction[1] = xto[1] - x[1][1];
    direction[2] = xto[2] - x[1][2];
    VECTOR_PBC(direction);
    
    double mol = direction[0]*direction[0] + direction[1]*direction[1] + direction[2]*direction[2];
    mol = sqrt(mol);
    direction[0] /= mol;
    direction[1] /= mol;
    direction[2] /= mol;
    
    x[1][0] += ref[0] * direction[0];
    x[1][1] += ref[0] * direction[1];
    x[1][2] += ref[0] * direction[2];
    
    // vector R(real)=R1-R2;
    
    for(int i=0; i<3; i++) 
      dx[i] = x[0][i]-x[1][i];
    VECTOR_PBC(dx);

    // h direction
    double h = 0.0;
    for(int i=0; i<3; i++) h+=dx[i]*direction[i];
    energy += 0.5*k[0]*h*h;
    // AWGL
    h_save = h; // save for extraction

    double dh[3],fh[3];
    for(int i=0; i<3; i++) 
    {
      dh[i]=direction[i]*h;
      fh[i]=-k[0]*dh[i];
      f[0][i]+=fh[i];
      f[1][i]-=fh[i];
      virial[i]=fh[i]*dh[i]*dh[i];
    }

    // r direction
    
    double r = 0.0;
    double dr[3],fr[3];
    for(int i=0; i<3; i++)
    {
      dr[i]=dx[i]-dh[i];
      r += dr[i]*dr[i];
    }
    energy += 0.5*k[1]*r;

    for(int i=0; i<3; i++) 
    {
      fr[i]=-k[1]*dr[i];
      f[0][i]+=fr[i];
      f[1][i]-=fr[i];
      virial[i]=fr[i]*dr[i]*dr[i];
    } 
    
    diff=h;
    dx[0]=r;
  }
  
  // Virial calculation
  virial[0] += dx[0]*dx[0]*ff[0];
  virial[1] += dx[1]*dx[1]*ff[1];
  virial[2] += dx[2]*dx[2]*ff[2];

  virial[0] /= comm->nprocs;
  virial[1] /= comm->nprocs;
  virial[2] /= comm->nprocs;
}

/* ---------------------------------------------------------------------- */

void FixUmbrella::init_group(int igroup)
{
  if(type_grp[igroup] == GTP_CG)
    for(int i = 0; i<natms[igroup]; i++) factor_list[igroup][i] = 1.0;
  
  if(type_grp[igroup] == GTP_CM)
  {
    double *_mass = new double[natms[igroup]];
	
	for(int i = 0; i<natms[igroup]; i++)
	{
	  _mass[i] = 0.0; 
	  factor_list[igroup][i] = 0.0;
	  int id = atom->map(tag_list[igroup][i]);
	  if(id>=0 && id<atom->nlocal) _mass[i] = atom->mass[atom->type[id]];
	}
	
	MPI_Allreduce(_mass, factor_list[igroup], natms[igroup], MPI_DOUBLE, MPI_SUM, world);
	
	double total_mass =0.0;
	for (int i=0; i<natms[igroup]; i++) total_mass += factor_list[igroup][i];
	for (int i=0; i<natms[igroup]; i++) factor_list[igroup][i] /= total_mass;
  
    delete [] _mass;
  }
  
  memory->create(r[igroup],natms[igroup],3,"FixUmbrella:r[igroup]");
  memory->create(_r[igroup],natms[igroup],3,"FixUmbrella:r[igroup]");
}

void FixUmbrella::setup_group(int igroup)
{
  int nlocal = atom->nlocal;
  int natm = natms[igroup];
  double **ax = atom->x;
  double **pos = r[igroup];
  double *factors = factor_list[igroup];
  double disp[3];
  
  memset(r[igroup][0],0,sizeof(double)*3*natm);
  memset(_r[igroup][0],0,sizeof(double)*3*natm);
  
  for(int i=0; i<natm; i++)
  {
	int _id;
	_id = id_list[igroup][i] = atom->map(tag_list[igroup][i]);
	if(_id<0 || _id>=nlocal) id_list[igroup][i]=-1;
	else memcpy(_r[igroup][i],ax[_id],sizeof(double)*3);
  }
  
  MPI_Allreduce(_r[igroup][0],pos[0],natm*3, MPI_DOUBLE, MPI_SUM, world);
  
  VECTOR_ZERO(x[igroup]);
 
  for(int i=1; i<natm; i++)
  {
    VECTOR_SUB(disp,pos[i],pos[0]);
	VECTOR_PBC(disp);
	VECTOR_SCALE_ADD(x[igroup],disp,factors[i]);
  }
  
  VECTOR_SELF_ADD(x[igroup],pos[0]);
  VECTOR_PBC(x[igroup]);
}

void FixUmbrella::cal_grp_force(int igroup)
{
  int natm = natms[igroup];
  int *ids = id_list[igroup];
  double *factors = factor_list[igroup];
  double **ff = atom->f;
  
  for(int i=0; i<natm; i++) if(ids[i]>=0)
  { VECTOR_SCALE_ADD(ff[ids[i]],f[igroup],factors[i]); }
}

/* ---------------------------------------------------------------------- */

void FixUmbrella::write_log()
{ 
  fprintf(OutFile,BIGINT_FORMAT "  %lf",update->ntimestep, energy);
  
  if(coord==COORD_CART)
  {
    if (di[0]) fprintf(OutFile,"  %lf  %lf  %lf",dx[0],x[0][0],x[1][0]);
    if (di[1]) fprintf(OutFile,"  %lf  %lf  %lf",dx[1],x[0][1],x[1][1]);
    if (di[2]) fprintf(OutFile,"  %lf  %lf  %lf",dx[2],x[0][2],x[1][2]);
  }
  else if(coord==COORD_SPHERICAL) 
    fprintf(OutFile,"  %lf  %lf  %lf  %lf  %lf  %lf  %lf",diff+ref[0],
      x[0][0],x[0][1],x[0][2],x[1][0],x[1][1],x[1][2]);
  else if(coord==COORD_PT) 
    fprintf(OutFile,"  %lf  %lf",diff+ref[0], dx[0]);
  
  fprintf(OutFile,"\n");
  fflush(OutFile);
}

/* ---------------------------------------------------------------------- */
/* AWGL : Additional functions for REUS external driver interface         */
/* ---------------------------------------------------------------------- */

void FixUmbrella::compute_bias_stuff_for_external(double *k_mod, double *ref_mod, double *xa0_mod)
{
  // ** This is a simplified version of compute intended for use with
  //    the REUS external driver ** //

  bias_energy = 0.0;

  // Get index
  for(int i=0; i<2; i++)
    if(type_grp[i] == GTP_CEC)
        {
          cplx[i] = fix_evb->Engine->all_complex[cplx_id[i]];
          for(int j=0; j<3; j++) if(di[j]) x[i][j] = cplx[i]->cec->r_cec[j];
        }

        else if(type_grp[i] == GTP_CECV2)
        {
          cplx[i] = fix_evb->Engine->all_complex[cplx_id[i]];
          for(int j=0; j<3; j++) if(di[j]) x[i][j] = cplx[i]->cec_v2->r_cec[j];
        }
        else if(type_grp[i]== GTP_ATOM)
        {
          atom_id[i] = atom->map(atom_tag[i]);
          int location = me;
          if (atom_id[i]<0 || atom_id[i]>=atom->nlocal) location = 0;
          else
          {
            x[i][0] = atom->x[atom_id[i]][0];
            x[i][1] = atom->x[atom_id[i]][1];
            x[i][2] = atom->x[atom_id[i]][2];
          }

          atom_rank[i]=0;
          MPI_Allreduce(&location,atom_rank+i,1,MPI_INT,MPI_SUM,world);
          MPI_Bcast(x[i],3,MPI_DOUBLE,atom_rank[i],world);
        }
        else if(type_grp[i]== GTP_CM) setup_group(i);

  dx[0] = dx[1] = dx[2] = 0.0;

  /******************************************/
  if(coord==COORD_CART) 
  {
    //for(int i=0; i<3; i++) if (di[i]) dx[i] = x[0][i]-x[1][i];
    for(int i=0; i<3; i++) if (di[i]) dx[i] = x[0][i]-xa0_mod[i];
    VECTOR_PBC(dx);
    //for(int i=0; i<3; i++) if (di[i]) dx[i] = dx[i]-ref[i];
    for(int i=0; i<3; i++) if (di[i]) dx[i] = dx[i]-ref_mod[i];
    VECTOR_PBC(dx);
	
    for(int i=0; i<3; i++) if (di[i])
    {
	  //ff[i] = -k[i];
	  ff[i] = -k_mod[i];
	  //f[0][i] = - k[i] * dx[i]; f[1][i] = -f[0][i];
	  f[0][i] = - k_mod[i] * dx[i]; f[1][i] = -f[0][i];
	  dx2[i] = dx[i]*dx[i];
	  //bias_energy += 0.5 * k[i] * dx2[i];
	  bias_energy += 0.5 * k_mod[i] * dx2[i];
    }

    //for(int i=0; i<3; i++) if (di[i]) dx[i] = x[0][i]-x[1][i];
    for(int i=0; i<3; i++) if (di[i]) dx[i] = x[0][i]-xa0_mod[i];
    VECTOR_PBC(dx);
  }
  /******************************************/
  if(coord==COORD_CYLINDER)
  { 
    int a0, a1;
    if(type_grp[0]==GTP_COORD) { a0=0; a1=1; }
    else { a1=0; a0=1; }

    // vector R(real)=R1-R2;
    //dx[0] = x[a1][0]-x[a0][0];
    //dx[1] = x[a1][1]-x[a0][1];
    //dx[2] = x[a1][2]-x[a0][2];
    dx[0] = x[a1][0]-xa0_mod[0];
    dx[1] = x[a1][1]-xa0_mod[1];
    dx[2] = x[a1][2]-xa0_mod[2];
    VECTOR_PBC(dx);

    // h direction
    double h = dx[0]*ref_mod[0] + dx[1]*ref_mod[1] + dx[2]*ref_mod[2];
    h_save = h; // save for extraction
    bias_energy += 0.5*k_mod[0]*h*h;

    double dh[3];
    dh[0] = ref_mod[0] * h;
    dh[1] = ref_mod[1] * h;
    dh[2] = ref_mod[2] * h;

    // r direction
    double dr[3];
    dr[0] = dx[0] - dh[0];
    dr[1] = dx[1] - dh[1];
    dr[2] = dx[2] - dh[2];
    double r = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
    bias_energy += 0.5*k_mod[1]*r;
  }
  /******************************************/

}

/* ---------------------------------------------------------------------- */

double FixUmbrella::compute_scalar()
{
  return bias_energy;
}

/* ---------------------------------------------------------------------- */

double FixUmbrella::compute_array(int i, int j)
{
  // Returns data from arrays
  double xx;
  if      (i == 0) xx = dx[j];
  else if (i == 1) xx = ref[j];
  else if (i == 2) xx = k[j];
  else if (i == 3) xx = h_save;
  else if (i == 4) {
    int a0, a1;
    if(type_grp[0]==GTP_COORD) { a0=0; a1=1; }
    else { a1=0; a0=1; }
    xx = x[a0][j];
  }
  return xx;
}

/* ---------------------------------------------------------------------- */

void FixUmbrella::modify_fix(int which, double *values, char *notused)
{
  // Sets a specified variable to the input value(s)
  if (which == 1) {
    ref[0] = values[0];
    ref[1] = values[1];
    ref[2] = values[2];
  }
  else if (which == 2) {
    k[0] = values[0];
    k[1] = values[1];
    k[2] = values[2];
  }
  else if (which == 4) {
    int a0, a1;
    if(type_grp[0]==GTP_COORD) { a0=0; a1=1; }
    else { a1=0; a0=1; }
    x[a0][0] = values[0];
    x[a0][1] = values[1];
    x[a0][2] = values[2];
  }
}

/* ---------------------------------------------------------------------- */
