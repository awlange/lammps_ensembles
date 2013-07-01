/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------
   Contributing author:  Axel Kohlmeyer (TempleU)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <errno.h>

#include "fix_colvars.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "respa.h"
#include "update.h"

#include "colvarproxy_lammps.h"
// awgl
#include "colvarbias.h"

/* re-usable integer hash table code with static linkage. */

/** hash table top level data structure */
typedef struct inthash_t {
  struct inthash_node_t **bucket;        /* array of hash nodes */
  int size;                           /* size of the array */
  int entries;                        /* number of entries in table */
  int downshift;                      /* shift cound, used in hash function */
  int mask;                           /* used to select bits for hashing */
} inthash_t;

/** hash table node data structure */
typedef struct inthash_node_t {
  int data;                           /* data in hash node */
  int key;                            /* key for hash lookup */
  struct inthash_node_t *next;        /* next node in hash chain */
} inthash_node_t;

#define HASH_FAIL  -1
#define HASH_LIMIT  0.5

/* initialize new hash table  */
static void inthash_init(inthash_t *tptr, int buckets);
/* lookup entry in hash table */
static int inthash_lookup(void *tptr, int key);
/* insert an entry into hash table. */
static int inthash_insert(inthash_t *tptr, int key, int data);
/* delete the hash table */
static void inthash_destroy(inthash_t *tptr);
/* adapted sort for in-place sorting of map indices. */
static void id_sort(int *idmap, int left, int right);

/************************************************************************
 * integer hash code:
 ************************************************************************/

/* inthash() - Hash function returns a hash number for a given key.
 * tptr: Pointer to a hash table, key: The key to create a hash number for */
static int inthash(const inthash_t *tptr, int key) {
  int hashvalue;

  hashvalue = (((key*1103515249)>>tptr->downshift) & tptr->mask);
  if (hashvalue < 0) {
    hashvalue = 0;
  }

  return hashvalue;
}

/*
 *  rebuild_table_int() - Create new hash table when old one fills up.
 *
 *  tptr: Pointer to a hash table
 */
static void rebuild_table_int(inthash_t *tptr) {
  inthash_node_t **old_bucket, *old_hash, *tmp;
  int old_size, h, i;

  old_bucket=tptr->bucket;
  old_size=tptr->size;

  /* create a new table and rehash old buckets */
  inthash_init(tptr, old_size<<1);
  for (i=0; i<old_size; i++) {
    old_hash=old_bucket[i];
    while(old_hash) {
      tmp=old_hash;
      old_hash=old_hash->next;
      h=inthash(tptr, tmp->key);
      tmp->next=tptr->bucket[h];
      tptr->bucket[h]=tmp;
      tptr->entries++;
    } /* while */
  } /* for */

  /* free memory used by old table */
  free(old_bucket);

  return;
}

/*
 *  inthash_init() - Initialize a new hash table.
 *
 *  tptr: Pointer to the hash table to initialize
 *  buckets: The number of initial buckets to create
 */
void inthash_init(inthash_t *tptr, int buckets) {

  /* make sure we allocate something */
  if (buckets==0)
    buckets=16;

  /* initialize the table */
  tptr->entries=0;
  tptr->size=2;
  tptr->mask=1;
  tptr->downshift=29;

  /* ensure buckets is a power of 2 */
  while (tptr->size<buckets) {
    tptr->size<<=1;
    tptr->mask=(tptr->mask<<1)+1;
    tptr->downshift--;
  } /* while */

  /* allocate memory for table */
  tptr->bucket=(inthash_node_t **) calloc(tptr->size, sizeof(inthash_node_t *));

  return;
}

/*
 *  inthash_lookup() - Lookup an entry in the hash table and return a pointer to
 *    it or HASH_FAIL if it wasn't found.
 *
 *  tptr: Pointer to the hash table
 *  key: The key to lookup
 */
int inthash_lookup(void *ptr, int key) {
  const inthash_t *tptr = (const inthash_t *) ptr;
  int h;
  inthash_node_t *node;


  /* find the entry in the hash table */
  h=inthash(tptr, key);
  for (node=tptr->bucket[h]; node!=NULL; node=node->next) {
    if (node->key == key)
      break;
  }

  /* return the entry if it exists, or HASH_FAIL */
  return(node ? node->data : HASH_FAIL);
}

/*
 *  inthash_insert() - Insert an entry into the hash table.  If the entry already
 *  exists return a pointer to it, otherwise return HASH_FAIL.
 *
 *  tptr: A pointer to the hash table
 *  key: The key to insert into the hash table
 *  data: A pointer to the data to insert into the hash table
 */
int inthash_insert(inthash_t *tptr, int key, int data) {
  int tmp;
  inthash_node_t *node;
  int h;

  /* check to see if the entry exists */
  if ((tmp=inthash_lookup(tptr, key)) != HASH_FAIL)
    return(tmp);

  /* expand the table if needed */
  while (tptr->entries>=HASH_LIMIT*tptr->size)
    rebuild_table_int(tptr);

  /* insert the new entry */
  h=inthash(tptr, key);
  node=(struct inthash_node_t *) malloc(sizeof(inthash_node_t));
  node->data=data;
  node->key=key;
  node->next=tptr->bucket[h];
  tptr->bucket[h]=node;
  tptr->entries++;

  return HASH_FAIL;
}

/*
 * inthash_destroy() - Delete the entire table, and all remaining entries.
 *
 */
void inthash_destroy(inthash_t *tptr) {
  inthash_node_t *node, *last;
  int i;

  for (i=0; i<tptr->size; i++) {
    node = tptr->bucket[i];
    while (node != NULL) {
      last = node;
      node = node->next;
      free(last);
    }
  }

  /* free the entire array of buckets */
  if (tptr->bucket != NULL) {
    free(tptr->bucket);
    memset(tptr, 0, sizeof(inthash_t));
  }
}

/************************************************************************
 * integer list sort code:
 ************************************************************************/

/* sort for integer map. initial call  id_sort(idmap, 0, natoms - 1); */
static void id_sort(int *idmap, int left, int right)
{
  int pivot, l_hold, r_hold;

  l_hold = left;
  r_hold = right;
  pivot = idmap[left];

  while (left < right) {
    while ((idmap[right] >= pivot) && (left < right))
      right--;
    if (left != right) {
      idmap[left] = idmap[right];
      left++;
    }
    while ((idmap[left] <= pivot) && (left < right))
      left++;
    if (left != right) {
      idmap[right] = idmap[left];
      right--;
    }
  }
  idmap[left] = pivot;
  pivot = left;
  left = l_hold;
  right = r_hold;

  if (left < pivot)
    id_sort(idmap, left, pivot-1);
  if (right > pivot)
    id_sort(idmap, pivot+1, right);
}

/***************************************************************/

using namespace LAMMPS_NS;
using namespace FixConst;

// initialize static class members
int FixColvars::instances=0;

/***************************************************************
 create class and parse arguments in LAMMPS script. Syntax:

 fix ID group-ID colvars <config_file> [optional flags...]

 optional keyword value pairs:

  input   <input prefix>    (for restarting/continuing, defaults to
                             NULL, but set to <output prefix> at end)
  output  <output prefix>   (defaults to 'out')
  seed    <integer>         (seed for RNG, defaults to '1966')
  tstat   <fix label>       (label of thermostatting fix)

 TODO: add (optional) arguments for RNG seed, temperature compute
 ***************************************************************/
FixColvars::FixColvars(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 4)
    error->all(FLERR,"Illegal fix colvars command: too few arguments");

  if (atom->tag_enable == 0)
    error->all(FLERR,"Cannot use fix colvars without atom IDs defined");

  if (atom->rmass_flag)
    error->all(FLERR,"Cannot use fix colvars for atoms with rmass attribute");

  if (instances)
    error->all(FLERR,"Only one fix colvars can be active at a time");
  ++instances;

  scalar_flag = 1;
  global_freq = 1;
  nevery = 1;
  extscalar = 1;

  me = comm->me;

  conf_file = strdup(arg[3]);
  rng_seed = 1966;

  inp_name = NULL;
  out_name = NULL;
  tmp_name = NULL;

  /* parse optional arguments */
  int argsdone = 4;
  while (argsdone+1 < narg) {
    if (0 == strcmp(arg[argsdone], "input")) {
      inp_name = strdup(arg[argsdone+1]);
    } else if (0 == strcmp(arg[argsdone], "output")) {
      out_name = strdup(arg[argsdone+1]);
    } else if (0 == strcmp(arg[argsdone], "seed")) {
      rng_seed = atoi(arg[argsdone+1]);
    } else if (0 == strcmp(arg[argsdone], "tstat")) {
      tmp_name = strdup(arg[argsdone+1]);
    } else {
      error->all(FLERR,"Unknown fix colvars parameter");
    }
    ++argsdone; ++argsdone;
  }

  if (!out_name) out_name = strdup("out");

  /* initialize various state variables. */
  tstat_id = -1;
  energy = 0.0;
  nlevels_respa = 0;
  num_coords = 0;
  coords = forces = oforce = NULL;
  comm_buf = NULL;
  force_buf = NULL;
  proxy = NULL;
  idmap = NULL;

  /* storage required to communicate a single coordinate or force. */
  size_one = sizeof(struct commdata);
}

/*********************************
 * Clean up on deleting the fix. *
 *********************************/
FixColvars::~FixColvars()
{
  memory->sfree(conf_file);
  memory->sfree(inp_name);
  memory->sfree(out_name);
  memory->sfree(tmp_name);
  deallocate();
  --instances;
}

/* ---------------------------------------------------------------------- */

void FixColvars::deallocate()
{
  memory->destroy(comm_buf);

  if (proxy) {
    delete proxy;
    inthash_t *hashtable = (inthash_t *)idmap;
    inthash_destroy(hashtable);
    delete hashtable;
  }

  proxy = NULL;
  idmap = NULL;
  coords = NULL;
  forces = NULL;
  oforce = NULL;
  comm_buf = NULL;
}

/* ---------------------------------------------------------------------- */

void FixColvars::post_run()
{
  deallocate();
  memory->sfree(inp_name);
  inp_name = strdup(out_name);
}

/* ---------------------------------------------------------------------- */

int FixColvars::setmask()
{
  int mask = 0;
  mask |= THERMO_ENERGY;
  mask |= MIN_POST_FORCE;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= POST_RUN;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

// initial setup of colvars run.

void FixColvars::init()
{
  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;


  int i,nme,tmp,ndata;

  MPI_Status status;
  MPI_Request request;

  // collect a list of atom type by atom id for the entire system.
  // the colvar module requires this information to set masses. :-(

  int *typemap,*type_buf;
  int nlocal_max,tag_max,max;
  const int * const tag  = atom->tag;
  const int * const type = atom->type;
  int nlocal = atom->nlocal;

  max=0;
  for (i = 0; i < nlocal; i++) max = MAX(max,tag[i]);
  MPI_Allreduce(&max,&tag_max,1,MPI_INT,MPI_MAX,world);
  MPI_Allreduce(&nlocal,&nlocal_max,1,MPI_INT,MPI_MAX,world);

  if (me == 0) {
    typemap = new int[tag_max+1];
    memset(typemap,0,sizeof(int)*tag_max);
  }
  type_buf = new int[2*nlocal_max];

  if (me == 0) {
    for (i=0; i<nlocal; ++i)
      typemap[tag[i]] = type[i];

    // loop over procs to receive and apply remote data

    for (i=1; i < comm->nprocs; ++i) {
      MPI_Irecv(type_buf, 2*nlocal_max, MPI_INT, i, 0, world, &request);
      MPI_Send(&tmp, 0, MPI_INT, i, 0, world);
      MPI_Wait(&request, &status);
      MPI_Get_count(&status, MPI_INT, &ndata);

      for (int k=0; k<ndata; k+=2)
        typemap[type_buf[k]] = type_buf[k+1];
    }
  } else { // me != 0

    // copy tag/type data into communication buffer

    nme = 0;
    for (i=0; i<nlocal; ++i) {
      type_buf[nme] = tag[i];
      type_buf[nme+1] = type[i];
      nme +=2;
    }
    /* blocking receive to wait until it is our turn to send data. */
    MPI_Recv(&tmp, 0, MPI_INT, 0, 0, world, &status);
    MPI_Rsend(type_buf, nme, MPI_INT, 0, 0, world);
  }

  // now create and initialize the colvar proxy

  if (me == 0) {

    if (inp_name) {
      if (strcmp(inp_name,"NULL") == 0) {
        memory->sfree(inp_name);
        inp_name = NULL;
      }
    }

    double t_target = 0.0;
    if (tmp_name) {
      if (strcmp(tmp_name,"NULL") == 0)
        tstat_id = -1;
      else {
        tstat_id = modify->find_fix(tmp_name);
        if (tstat_id < 0) error->one(FLERR,"Could not find tstat fix ID");
        double *tt = (double*)modify->fix[tstat_id]->extract("t_target",tmp);
        if (tt) t_target = *tt;
      }
    }

    proxy = new colvarproxy_lammps(lmp,conf_file,inp_name,out_name,
                                   rng_seed,t_target,typemap);
    coords = proxy->get_coords();
    forces = proxy->get_forces();
    oforce = proxy->get_oforce();
    num_coords = coords->size();
  }

  // send the list of all colvar atom IDs to all nodes.
  // also initialize and build hashtable on master.

  MPI_Bcast(&num_coords, 1, MPI_INT, 0, world);
  memory->create(taglist,num_coords,"colvars:taglist");
  memory->create(force_buf,3*num_coords,"colvars:force_buf");

  if (me == 0) {
    std::vector<int> *tags_list = proxy->get_tags();
    std::vector<int> &tl = *tags_list;
    inthash_t *hashtable=new inthash_t;
    inthash_init(hashtable, num_coords);
    idmap = (void *)hashtable;

    for (i=0; i < num_coords; ++i) {
      taglist[i] = tl[i];
      inthash_insert(hashtable, tl[i], i);
    }
  }
  MPI_Bcast(taglist, num_coords, MPI_INT, 0, world);

  // determine size of comm buffer
  nme=0;
  for (i=0; i < num_coords; ++i) {
    const int k = atom->map(taglist[i]);
    if ((k >= 0) && (k < nlocal))
      ++nme;
  }

  MPI_Allreduce(&nme,&nmax,1,MPI_INT,MPI_MAX,world);
  memory->create(comm_buf,nmax,"colvars:comm_buf");

  const double * const * const x = atom->x;

  if (me == 0) {

    std::vector<struct commdata> &cd = *coords;
    std::vector<struct commdata> &of = *oforce;

    // store coordinate data in holding array, clear old forces

    for (i=0; i<num_coords; ++i) {
      const int k = atom->map(taglist[i]);
      if ((k >= 0) && (k < nlocal)) {
        of[i].tag  = cd[i].tag  = tag[k];
        of[i].type = cd[i].type = type[k];
        cd[i].x = x[k][0];
        cd[i].y = x[k][1];
        cd[i].z = x[k][2];
        of[i].x = of[i].y = of[i].z = 0.0;
      }
    }

    // loop over procs to receive and apply remote data

    for (i=1; i < comm->nprocs; ++i) {
      int maxbuf = nmax*size_one;
      MPI_Irecv(comm_buf, maxbuf, MPI_BYTE, i, 0, world, &request);
      MPI_Send(&tmp, 0, MPI_INT, i, 0, world);
      MPI_Wait(&request, &status);
      MPI_Get_count(&status, MPI_BYTE, &ndata);
      ndata /= size_one;

      for (int k=0; k<ndata; ++k) {
        const int j = inthash_lookup(idmap, comm_buf[k].tag);
        if (j != HASH_FAIL) {
          of[j].tag  = cd[j].tag  = comm_buf[k].tag;
          of[j].type = cd[j].type = comm_buf[k].type;
          cd[j].x = comm_buf[k].x;
          cd[j].y = comm_buf[k].y;
          cd[j].z = comm_buf[k].z;
          of[j].x = of[j].y = of[j].z = 0.0;
        }
      }
    }
  } else { // me != 0

    // copy coordinate data into communication buffer

    nme = 0;
    for (i=0; i<num_coords; ++i) {
      const int k = atom->map(taglist[i]);
      if ((k >= 0) && (k < nlocal)) {
        comm_buf[nme].tag  = tag[k];
        comm_buf[nme].type = type[k];
        comm_buf[nme].x    = x[k][0];
        comm_buf[nme].y    = x[k][1];
        comm_buf[nme].z    = x[k][2];
        ++nme;
      }
    }
    /* blocking receive to wait until it is our turn to send data. */
    MPI_Recv(&tmp, 0, MPI_INT, 0, 0, world, &status);
    MPI_Rsend(comm_buf, nme*size_one, MPI_BYTE, 0, 0, world);
  }

  // clear temporary storage
  if (me == 0) delete typemap;
  delete type_buf;
}

/* ---------------------------------------------------------------------- */

void FixColvars::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else
    post_force_respa(vflag,0,0);
}

/* ---------------------------------------------------------------------- */
/* Main colvars handler:
 * Send coodinates and add colvar forces to atoms. */
void FixColvars::post_force(int vflag)
{
  // some housekeeping: update status of the proxy as needed.
  if (me == 0) {
    if (proxy->want_exit())
      error->one(FLERR,"Run halted on request from colvars module.\n");

    if (tstat_id < 0) {
      proxy->set_temperature(0.0);
    } else {
      int tmp;
      // get thermostat target temperature from corresponding fix,
      // if the fix supports extraction.
      double *tt = (double *) modify->fix[tstat_id]->extract("t_target",tmp);
      if (tt)
        proxy->set_temperature(*tt);
      else
        proxy->set_temperature(0.0);
    }
  }

  const int * const tag = atom->tag;
  const double * const * const x = atom->x;
  double * const * const f = atom->f;
  const int nlocal = atom->nlocal;

  /* check and potentially grow local communication buffers. */
  int i,nmax_new,nme=0;
  for (i=0; i < num_coords; ++i) {
    const int k = atom->map(taglist[i]);
    if ((k >= 0) && (k < nlocal))
      ++nme;
  }

  MPI_Allreduce(&nme,&nmax_new,1,MPI_INT,MPI_MAX,world);
  if (nmax_new > nmax) {
    nmax = nmax_new;
    memory->grow(comm_buf,nmax,"colvars:comm_buf");
  }

  MPI_Status status;
  MPI_Request request;
  int tmp, ndata;

  if (me == 0) {
    std::vector<struct commdata> &cd = *coords;

    // store coordinate data

    for (i=0; i<num_coords; ++i) {
      const int k = atom->map(taglist[i]);
      if ((k >= 0) && (k < nlocal)) {
        cd[i].x = x[k][0];
        cd[i].y = x[k][1];
        cd[i].z = x[k][2];
      }
    }

    /* loop over procs to receive remote data */
    for (i=1; i < comm->nprocs; ++i) {
      int maxbuf = nmax*size_one;
      MPI_Irecv(comm_buf, maxbuf, MPI_BYTE, i, 0, world, &request);
      MPI_Send(&tmp, 0, MPI_INT, i, 0, world);
      MPI_Wait(&request, &status);
      MPI_Get_count(&status, MPI_BYTE, &ndata);
      ndata /= size_one;

      for (int k=0; k<ndata; ++k) {
        const int j = inthash_lookup(idmap, comm_buf[k].tag);
        if (j != HASH_FAIL) {
          cd[j].x = comm_buf[k].x;
          cd[j].y = comm_buf[k].y;
          cd[j].z = comm_buf[k].z;
        }
      }
    }

  } else { // me != 0
    /* copy coordinate data into communication buffer */
    nme = 0;
    for (i=0; i<num_coords; ++i) {
      const int k = atom->map(taglist[i]);
      if ((k >= 0) && (k < nlocal)) {
        comm_buf[nme].tag = tag[k];
        comm_buf[nme].x = x[k][0];
        comm_buf[nme].y = x[k][1];
        comm_buf[nme].z = x[k][2];
        ++nme;
      }
    }
    /* blocking receive to wait until it is our turn to send data. */
    MPI_Recv(&tmp, 0, MPI_INT, 0, 0, world, &status);
    MPI_Rsend(comm_buf, nme*size_one, MPI_BYTE, 0, 0, world);
  }

  ////////////////////////////////////////////////////////////////////////
  // call our workhorse and retrieve additional information.
  if (me == 0) {
    energy = proxy->compute();
    store_forces = proxy->need_system_forces();
  }
  ////////////////////////////////////////////////////////////////////////

  // broadcast store_forces flag and energy data to all processors
  MPI_Bcast(&energy, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&store_forces, 1, MPI_INT, 0, world);

  // broadcast and apply biasing forces

  if (me == 0) {
    std::vector<struct commdata> &fo = *forces;
    double *fbuf = force_buf;
    for (int j=0; j < num_coords; ++j) {
      *fbuf++ = fo[j].x;
      *fbuf++ = fo[j].y;
      *fbuf++ = fo[j].z;
    }
  }
  MPI_Bcast(force_buf, 3*num_coords, MPI_DOUBLE, 0, world);

  for (int i=0; i < num_coords; ++i) {
    const int k = atom->map(taglist[i]);
    if ((k >= 0) && (k < nlocal)) {
      f[k][0] += force_buf[3*i+0];
      f[k][1] += force_buf[3*i+1];
      f[k][2] += force_buf[3*i+2];
    }
  }
}

/* ---------------------------------------------------------------------- */
void FixColvars::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */
void FixColvars::post_force_respa(int vflag, int ilevel, int iloop)
{
  /* only process colvar forces on the outmost RESPA level. */
  if (ilevel == nlevels_respa-1) post_force(vflag);
  return;
}

/* ---------------------------------------------------------------------- */
void FixColvars::end_of_step()
{
  if (store_forces) {

    const int * const tag = atom->tag;
    double * const * const f = atom->f;
    const int nlocal = atom->nlocal;

    /* check and potentially grow local communication buffers. */
    int i,nmax_new,nme=0;
    for (i=0; i < num_coords; ++i) {
      const int k = atom->map(taglist[i]);
      if ((k >= 0) && (k < nlocal))
        ++nme;
    }

    MPI_Allreduce(&nme,&nmax_new,1,MPI_INT,MPI_MAX,world);
    if (nmax_new > nmax) {
      nmax = nmax_new;
      memory->grow(comm_buf,nmax,"colvars:comm_buf");
    }

    MPI_Status status;
    MPI_Request request;
    int tmp, ndata;

    if (me == 0) {

      // store old force data
      std::vector<struct commdata> &of = *oforce;

      for (i=0; i<num_coords; ++i) {
        const int k = atom->map(taglist[i]);
        if ((k >= 0) && (k < nlocal)) {

          const int j = inthash_lookup(idmap, tag[k]);
          if (j != HASH_FAIL) {
            of[j].x = f[k][0];
            of[j].y = f[k][1];
            of[j].z = f[k][2];
          }
        }
      }

      /* loop over procs to receive remote data */
      for (i=1; i < comm->nprocs; ++i) {
        int maxbuf = nmax*size_one;
        MPI_Irecv(comm_buf, maxbuf, MPI_BYTE, i, 0, world, &request);
        MPI_Send(&tmp, 0, MPI_INT, i, 0, world);
        MPI_Wait(&request, &status);
        MPI_Get_count(&status, MPI_BYTE, &ndata);
        ndata /= size_one;

        for (int k=0; k<ndata; ++k) {
          const int j = inthash_lookup(idmap, comm_buf[k].tag);
          if (j != HASH_FAIL) {
            of[j].x = comm_buf[k].x;
            of[j].y = comm_buf[k].y;
            of[j].z = comm_buf[k].z;
          }
        }
      }

    } else { // me != 0
      /* copy total force data into communication buffer */
      nme = 0;
      for (i=0; i<num_coords; ++i) {
        const int k = atom->map(taglist[i]);
        if ((k >= 0) && (k < nlocal)) {
          comm_buf[nme].tag  = tag[k];
          comm_buf[nme].x    = f[k][0];
          comm_buf[nme].y    = f[k][1];
          comm_buf[nme].z    = f[k][2];
          ++nme;
        }
      }
      /* blocking receive to wait until it is our turn to send data. */
      MPI_Recv(&tmp, 0, MPI_INT, 0, 0, world, &status);
      MPI_Rsend(comm_buf, nme*size_one, MPI_BYTE, 0, 0, world);
    }
  }
}

/* ---------------------------------------------------------------------- */

double FixColvars::compute_scalar()
{
  //return energy; // normal
  // AWGL: modified for LAMMPS ensembles
  if (me == 0) {
    energy = proxy->compute();
  }
  MPI_Bcast(&energy, 1, MPI_DOUBLE, 0, world);
  return energy; 
}

/* ---------------------------------------------------------------------- */
/* local memory usage. approximately. */
double FixColvars::memory_usage(void)
{
  double bytes = (double) (num_coords * (2*sizeof(int)+3*sizeof(double)));
  bytes += (double) (nmax*size_one) + sizeof(this);
  return bytes;
}

/* ---------------------------------------------------------------------- */
// awgl
void FixColvars::modify_fix(int which, double *var, char *filename) 
{
  if (me != 0) return;

  // Only one bias for now
  colvarbias *b = proxy->colvars->biases[0];

  if (which == 0) {
    // extract the colvar_center value
    var[0] = b->extract_value(0);
    var[1] = b->extract_value(1);
  }
  else if (which == 1) {
    // change the colvar_center value
    std::ostringstream os0, os1;
    os0 << var[0]; 
    os1 << var[1]; 
    std::string str = "";
    str.append("forceConstant " + os0.str() + "\n");
    str.append("centers " + os1.str() + "\n");
    b->change_configuration(str); 
  }
  else if (which == 3) {
    // close the output file and make append true for re-opening
    proxy->colvars->cv_traj_os.close();
    proxy->colvars->cv_traj_append = true;
  }
  else if (which == 4) {
    // get current output file name
    filename[proxy->colvars->cv_traj_name.size()] = 0; // to NULL terminate
    memcpy(filename, proxy->colvars->cv_traj_name.c_str(), proxy->colvars->cv_traj_name.size());
  }
  else if (which == 5) {
    // change output file name
    proxy->colvars->cv_traj_name.clear();
    proxy->colvars->cv_traj_name.assign(filename);
  }
  else { 
    printf("invalid which option for modify_fix.\n");
  }
}
