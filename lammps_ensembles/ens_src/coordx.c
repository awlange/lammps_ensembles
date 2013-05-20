/*
 * coord_exchange.c
 * This file is part of lammps-ensembles
 *
 * Copyright (C) 2012 - Mladen Rasic & Luke Westby
 *
 * lammps-ensembles is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * lammps-ensembles is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with lammps-ensembles; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, 
 * Boston, MA  02110-1301  USA
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "library.h"
#include "replica.h"

/*
 *  AWGL : This function performs coordinate exchanges for a given list of neighbors for each
 *         replica. This should be general for any restraint, temperature, etc. defined. The neighbor
 *         list should also support arbitrary topologies. Finally, we also want to support any number
 *         of dimensions for exchanging.
 */

// --- Helper function --- //
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

// --- Main function --- //
void coord_exchange(void *lmp, MPI_Comm subcomm, int ncomms, int comm, 
                    Replica *this_replica, char* fix, int sseed) 
{

/*----------------------------------------------------------------------------------
 * MPI things
 */
    
    // who i am in subcomm, how many in subcomm
    int this_local_proc, n_procs;
    MPI_Comm_rank(subcomm, &this_local_proc);
    MPI_Comm_size(subcomm, &n_procs);

    // who i am in COMM_WORLD
    int this_global_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &this_global_proc);

#ifdef COORDX_DEBUG_LEVEL_TWO
    if (this_global_proc == 0) printf("Now in coordx.c\n");
#endif

/*----------------------------------------------------------------------------------
 * create communicator for roots on subcomms
 */

    int split_key;                                          // color argument
    MPI_Comm roots;                                         // newcomm argument
    if(this_local_proc == 0) split_key = 0;                 // grab roots on subcomm
    else split_key = 1;
    MPI_Comm_split(MPI_COMM_WORLD, split_key, 0, &roots);

/*----------------------------------------------------------------------------------
 * grab info from args
 */

    MPI_Barrier(MPI_COMM_WORLD);
    int i;
    int i_comm    = comm;           // instance ID
    int i_ncomms  = ncomms;         // number of instances

    this_replica->comm = i_comm;                      // instance ID for this replica
    int i_replica_id   = this_replica->id;            // replica ID
    double i_temp      = this_replica->temperature;   // initial temperature for this proc
    int i_ndimensions  = this_replica->N_dimensions;  // number of dimensions
    int i_temp_dim     = this_replica->temp_dim;      // which dimension, if any, is the temperature swapping dimension 
    int i_nsteps = 0;                                 // total number of steps to run
    int i_nswaps = 0;                                 // total number of swaps for entire run
    int i_nevery = 0;                                 // frequency of swap, will depend on dimension
    for (i=0; i<i_ndimensions; ++i) {
      i_nsteps += this_replica->dim_run[i];
      i_nswaps += this_replica->dim_run[i] / this_replica->dim_nevery[i];
    }
    if (this_global_proc == 0) {
      printf("Total MD steps to take:  %d\n", i_nsteps);
      printf("Total swap runs to make: %d\n", i_nswaps);
    }

/*----------------------------------------------------------------------------------
 * setup update and initialize lammps
 */
 
    lammps_mod_inst(lmp, 4, NULL, "setup", &i_nsteps);
    lammps_mod_inst(lmp, 0, NULL, "init", NULL);


/*----------------------------------------------------------------------------------
 * grab boltzmann constant from lammps - depends on user "units" command
 */

    double *boltz_ptr = (double *)lammps_extract_global(lmp, "boltz");
    double boltz = *boltz_ptr;


/*----------------------------------------------------------------------------------
 * set up rngs
 */

    int i_sseed = sseed;
    int ranswapflag = 1;
    if(i_sseed == 0) ranswapflag = 0; 

    // Set up random number
    if (i_sseed < 0) {
      // If it's negative, the global root decides from the current time, 
      // broadcasts, and other ranks add their global rank. All seeds should be unique.
      int time_seed;
      if (this_global_proc == 0)
        time_seed = rng2_get_time_seed();
      MPI_Bcast(&time_seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
      rng2_seed(time_seed + this_global_proc);
#ifdef COORDX_DEBUG
      if(this_global_proc == 0) printf("Time seed = %d\n", time_seed);
#endif
    }
    else {
      rng2_seed(i_sseed + this_global_proc);
    }

/*----------------------------------------------------------------------------------
 * create lookup tables for useful values
 */

    // world2root - return global proc of root proc on a given subcomm
    int *world2root = (int *)malloc(sizeof(int) * i_ncomms);
    if(this_local_proc == 0)
        MPI_Allgather(&this_global_proc, 1, MPI_INT, world2root, 1, MPI_INT, roots);
    MPI_Bcast(world2root, i_ncomms, MPI_INT, 0, subcomm);
    
    // replicaid2temp - return given temperature for a given replica id
    double *replicaid2temp = (double *)malloc(sizeof(double) * i_ncomms);
    if(this_local_proc == 0)
        MPI_Allgather(&i_temp, 1, MPI_DOUBLE, replicaid2temp, 1, MPI_DOUBLE, roots);
    MPI_Bcast(replicaid2temp, i_ncomms, MPI_DOUBLE, 0, subcomm);

    // replicaid2world - return subcomm for a given replica_id
    int *world2replicaid = (int *)malloc(sizeof(int) * i_ncomms);
    int *replicaid2world = (int *)malloc(sizeof(int) * i_ncomms);
    if(this_local_proc == 0){
        MPI_Allgather(&i_replica_id, 1, MPI_INT, world2replicaid, 1, MPI_INT, roots);
        for (i = 0; i < i_ncomms; i++) replicaid2world[world2replicaid[i]] = i;
    }
    MPI_Bcast(replicaid2world, i_ncomms, MPI_INT, 0, subcomm);
    MPI_Bcast(world2replicaid, i_ncomms, MPI_INT, 0, subcomm);

    // world2temp - return the temperature for a given world -> should remain same throughout simulation 
    double *world2temp = (double *)malloc(sizeof(double) * i_ncomms);
    if (this_local_proc == 0) {
      MPI_Allgather(&this_replica->temperature, 1, MPI_DOUBLE, world2temp, 1, MPI_DOUBLE, roots);
    }
    MPI_Bcast(world2temp, i_ncomms, MPI_DOUBLE, 0, subcomm);

    if (this_global_proc == 0) {
      printf("Temperatures in each subcomm (static):\n");
      for (i=0; i<i_ncomms; ++i) {
        printf("%d : %f\n", i, world2temp[i]); 
      }
    }

/*----------------------------------------------------------------------------------
 * initialize integrator
 */

    lammps_mod_inst(lmp, 3, NULL, "setup", NULL);

/*----------------------------------------------------------------------------------
 *  Set up atom arrays for swapping coordinates and velocities 
 */

    int natoms = lammps_get_natoms(lmp);
    double *my_coords      = (double *)malloc(sizeof(double) * 3*natoms); 
    double *partner_coords = (double *)malloc(sizeof(double) * 3*natoms); 
    double *my_velocs      = (double *)malloc(sizeof(double) * 3*natoms); 
    double *partner_velocs = (double *)malloc(sizeof(double) * 3*natoms); 

/*----------------------------------------------------------------------------------
 * Swap statistics 
 */
   double TotalLoopTime = 0.0;
   int n_swaps_attempted;
   int n_swaps_successful;
   double acceptance_ratio;
   double average_acceptance_ratio = 0.0;
   int n_total_swaps_attempted  = 0;
   int n_total_swaps_successful = 0;

/*----------------------------------------------------------------------------------
 * print status header
 */

    if(this_global_proc == 0) {
      bigint *current_ptr = (bigint *)lammps_extract_global(lmp, "ntimestep");
      bigint timestep = *current_ptr;
      printf("---------------------------------\n");
      printf("Timestep %ld : Swap dimension %d Steps %d\n", 
              (long)timestep, 0, this_replica->dim_nevery[0]);
      int s;
      for (s=0; s<i_ncomms; ++s) {
        printf("  Subcomm %d : id = %d \n", s, world2replicaid[s]);
      }
    }

/*----------------------------------------------------------------------------------
 * main loops: 
 * Outer loop over total number of swaps
 * Inner loop over dimensions, go through each dimension sequentially in order 
 */

    int iswap, swap, dir, p_replica_id, partner_proc, partner_comm;
    int idim;
    double pe;                // My current potential energy
    double pe_partner;        // Partner's current potential energy
    double pe_swap;           // My potential energy if I have my partner's coordinates
    double pe_partner_swap;   // Partner's potential energy if has my coordinates 
    double *pe_ptr;
    double bias;              // My current bias energy
    double bias_partner;      // Partner's current bias energy
    double bias_swap;         // My bias energy if I have my partner's coordinates
    double bias_partner_swap; // Partner's bias energy if has my coordinates 
    double delta;
    // Same as above but potential energy + bias
    double pe_bias;              
    double pe_bias_partner;      
    double pe_bias_swap;         
    double pe_bias_partner_swap; 
    MPI_Status status;

    int n_swap_runs_completed = 0;

    // Loop until all swap runs are done
    while (n_swap_runs_completed < i_nswaps) {

      iswap = n_swap_runs_completed / i_ndimensions;

      for (idim=0; idim < i_ndimensions; ++idim) {

        // Timer
        MPI_Barrier(MPI_COMM_WORLD);
        double LoopTime_start = MPI_Wtime();

        /*-------------------------------------------------------------------------*/
        // 0. Set data for this dimension
        i_nevery = this_replica->dim_nevery[idim];

        /*-------------------------------------------------------------------------*/
        // 1. run for one period of timesteps for this dimension
        lammps_mod_inst(lmp, 2, "thermo_pe", "addstep", &i_nevery);
        lammps_mod_inst(lmp, 3, NULL, "run", &i_nevery);

        /*-------------------------------------------------------------------------*/
        // 2. extract potential energy from compute instance, and get my bias energy
        pe_ptr  = (double *)lammps_extract_compute(lmp, "thermo_pe", 0, 0);
        pe      = *pe_ptr;
        bias    = get_bias(lmp, fix, 0);
        pe_bias = pe + bias;

        /*-------------------------------------------------------------------------*/
        // 3. determine which direction matching takes place
        // -1 = negative direction along dimension
        //  1 = positive direction along dimension
        if (this_global_proc == 0) {
          if (ranswapflag == 0) {
            if (iswap % 2) dir = -1;
            else           dir =  1;
          }
          else if (rng2() < 0.5) dir = -1;
          else                   dir =  1;
        }
        MPI_Bcast(&dir, 1, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef COORDX_DEBUG
        if (this_global_proc == 0) printf("Swap dir = %d\n", dir); 
#endif

        /*-------------------------------------------------------------------------*/
        // 4. match partners
        partner_comm = -1;
        int dim_num = this_replica->dim_num[idim];
        if (dir == -1) { 
          if (dim_num % 2) partner_comm = this_replica->neighbors[2*idim+1];
          else             partner_comm = this_replica->neighbors[2*idim  ];
        } else {           
          if (dim_num % 2) partner_comm = this_replica->neighbors[2*idim  ];
          else             partner_comm = this_replica->neighbors[2*idim+1];
        }
        p_replica_id = -1;
        if (partner_comm >= 0) p_replica_id = world2replicaid[partner_comm];

        /*-------------------------------------------------------------------------*/
        // 5. get partner global rank
        partner_proc = -1;
        if(partner_comm >= 0 && partner_comm < i_ncomms) {
          partner_proc = world2root[partner_comm];
        }
#ifdef COORDX_DEBUG_LEVEL_TWO
        if(this_local_proc == 0) {
          printf("subcomm %d: replica_id %d p_replica_id = %d partner_comm = %d partner_proc = %d dim_num %d\n", 
                 i_comm, world2replicaid[i_comm], p_replica_id, partner_comm, partner_proc, dim_num);
        }
#endif

        /*-------------------------------------------------------------------------*/
        // 6. figure out if swap is okay
        swap = 0;
        n_swaps_attempted = 0;
        n_swaps_successful = 0;
        if (idim == i_temp_dim) {
          // *** Temperature swapping dimension *** //
          if (partner_proc != -1) {
            if (this_local_proc == 0) {
                n_swaps_attempted = 1;
                if (this_global_proc > partner_proc) {
                  // higher proc sends pe to lower proc
                  MPI_Send(&pe_bias, 1, MPI_DOUBLE, partner_proc, 0, MPI_COMM_WORLD);
                } else {
                  // lower proc recieves
                  MPI_Recv(&pe_bias_partner, 1, MPI_DOUBLE, partner_proc, 0, MPI_COMM_WORLD, &status);
                }
                // lower proc does calculations
                if (this_global_proc < partner_proc) {
                    delta = (pe_bias_partner - pe_bias) * 
                             (1.0 / (boltz * world2temp[partner_comm]) -
                              1.0 / (boltz * world2temp[i_comm]));
                    double my_rand = rng2();
                    // make decision monte carlo style
                    if (delta >= 0.0) swap = 1; // criterion of e^0 or greater -> probability of 1
                    else if (my_rand < exp(delta)) swap = 1; 
#ifdef COORDX_DEBUG
                    double prob = exp(delta);
                    if (delta >= 0.0) prob = 1.0;
                    printf("%d<-->%d pe_bias: %lf p_pe_bias: %lf  delta: %lf, probability: %f, rand: %lf swap: %d\n", 
                            i_comm, partner_comm, pe_bias, pe_bias_partner, delta, prob, my_rand, swap);
#endif
                }
                // send decision to higher proc
                if (this_global_proc < partner_proc) {
                    MPI_Send(&swap, 1, MPI_INT, partner_proc, 0, MPI_COMM_WORLD);
                } else {
                    MPI_Recv(&swap, 1, MPI_INT, partner_proc, 0, MPI_COMM_WORLD, &status);
                }
            }
          } 
        }
        else {
          // *** Hamiltonian swapping dimensions *** //
          if (partner_proc != -1) {
            // Get my coordinates and velocities 
            lammps_gather_atoms(lmp, "x", 1, 3, my_coords);
            lammps_gather_atoms(lmp, "v", 1, 3, my_velocs);
            if (this_local_proc == 0) {
              // Send my coordinates to my partner, recieve coordinates from my partner
              MPI_Sendrecv(my_coords,      3*natoms, MPI_DOUBLE, partner_proc, 0,
                           partner_coords, 3*natoms, MPI_DOUBLE, partner_proc, 0, MPI_COMM_WORLD, &status);
              MPI_Sendrecv(my_velocs,      3*natoms, MPI_DOUBLE, partner_proc, 0,
                           partner_velocs, 3*natoms, MPI_DOUBLE, partner_proc, 0, MPI_COMM_WORLD, &status);
            }
            MPI_Bcast(partner_coords, 3*natoms, MPI_DOUBLE, 0, subcomm);
            MPI_Bcast(partner_velocs, 3*natoms, MPI_DOUBLE, 0, subcomm);
            // Set coordinates temporarily for energy computation
            lammps_scatter_atoms(lmp, "x", 1, 3, partner_coords);
            lammps_scatter_atoms(lmp, "v", 1, 3, partner_velocs);
            lammps_mod_inst(lmp, 3, NULL, "setup_minimal", NULL);

            // * Compute PE with my partner's coordinates * //
            // Currently, this line REQUIRES that the input script contains the line "compute pe all pe"
            // While this is not a necessary step for CV swaps, it is here for generality in order to allow
            // the user to do Hamiltonian exchange without colvars. 
            pe_ptr  = (double *)lammps_extract_compute(lmp, "pe", 0, 0);
            pe_swap = *pe_ptr;
            
            // Compute my bias with partner's coordinates (note: this is skipped if fix="none")
            bias_swap    = get_bias(lmp, fix, 0);
            pe_bias_swap = pe_swap + bias_swap;
#ifdef COORDX_DEBUG_LEVEL_TWO
            printf("subcomm %d : bias %f bias_swap %f\n", i_comm, bias, bias_swap);
#endif

            if (this_local_proc == 0) {
                n_swaps_attempted = 1;
                if (this_global_proc > partner_proc) {
                  // higher proc sends pe_swap to lower proc
                  MPI_Send(&pe_bias,      1, MPI_DOUBLE, partner_proc, 0, MPI_COMM_WORLD);
                  MPI_Send(&pe_bias_swap, 1, MPI_DOUBLE, partner_proc, 0, MPI_COMM_WORLD);
                } else {
                  // lower proc recieves
                  MPI_Recv(&pe_bias_partner,      1, MPI_DOUBLE, partner_proc, 0, MPI_COMM_WORLD, &status);
                  MPI_Recv(&pe_bias_partner_swap, 1, MPI_DOUBLE, partner_proc, 0, MPI_COMM_WORLD, &status);
                }
                // lower proc does calculations
                if (this_global_proc < partner_proc) {
                    delta  = (pe_bias_swap + pe_bias_partner_swap) - (pe_bias + pe_bias_partner);
                    delta /= boltz * world2temp[i_comm];
                    double my_rand = rng2();
                    // make decision monte carlo style
                    if (delta <= 0.0) swap = 1; // criterion of e^0 or greater -> probability of 1
                    else if (my_rand < exp(-delta)) swap = 1; 
#ifdef COORDX_DEBUG
                    double prob = exp(-delta);
                    if (delta <= 0.0) prob = 1.0;
                    printf("Swap %d : %d<-->%d pe_b: %lf p_pe_b: %lf pe_b_swap: %lf p_pe_b_swap: %lf delta: %lf, probability: %f, rand: %lf swap: %d\n", 
                            iswap, i_comm, partner_comm, pe_bias, pe_bias_partner, pe_bias_swap, pe_bias_partner_swap, 
                            delta, prob, my_rand, swap);
#endif
                }
                // send decision to higher proc
                if (this_global_proc < partner_proc) {
                    MPI_Send(&swap, 1, MPI_INT, partner_proc, 0, MPI_COMM_WORLD);
                } else {
                    MPI_Recv(&swap, 1, MPI_INT, partner_proc, 0, MPI_COMM_WORLD, &status);
                }
            }
          } 
        }

        // broadcast decision to subcomm
        MPI_Bcast(&swap, 1, MPI_INT, 0, subcomm);

        /*-------------------------------------------------------------------------*/
        // 7. perform swap, if needed
        if (swap == 1) {
          // swap replica_id
          i_replica_id = p_replica_id;
          this_replica->id = i_replica_id;
          n_swaps_successful = 1; // for stats
        }
        if (swap == 1 && idim == i_temp_dim) {
          // *** Temperature swapping *** //
          lammps_gather_atoms(lmp, "x", 1, 3, my_coords);
          if (this_local_proc == 0) {
            MPI_Sendrecv(my_coords,      3*natoms, MPI_DOUBLE, partner_proc, 0,
                         partner_coords, 3*natoms, MPI_DOUBLE, partner_proc, 0, MPI_COMM_WORLD, &status);
          }
          MPI_Bcast(partner_coords, 3*natoms, MPI_DOUBLE, 0, subcomm);
          lammps_scatter_atoms(lmp, "x", 1, 3, partner_coords);
          // Recompute energy/force with my coordinates
          lammps_mod_inst(lmp, 3, NULL, "setup_minimal", NULL);
          get_bias(lmp, fix, 0);
          // scale velocities to match new temperature
          lammps_scale_velocities(lmp, world2temp[i_comm], world2temp[partner_comm]);
        }
        else if (swap == 0 && idim != i_temp_dim && partner_proc != -1) {
          // *** Hamiltonian swapping *** // 
          // Coordinates were swapped above in attempt but exchange failed. So, restore my coordinates. 
          lammps_scatter_atoms(lmp, "x", 1, 3, my_coords);
          lammps_scatter_atoms(lmp, "v", 1, 3, my_velocs);
          // Recompute energy/force with my coordinates
          lammps_mod_inst(lmp, 3, NULL, "setup_minimal", NULL);
          get_bias(lmp, fix, 0);
        }
        // swap stats
        if (this_local_proc == 0) {
          int sbufi = n_swaps_successful;
          int rbufi;
          MPI_Reduce(&sbufi, &rbufi, 1, MPI_INT, MPI_SUM, 0, roots);
          n_swaps_successful = rbufi / 2; // Divide by two b/c both from pairs added
          sbufi = n_swaps_attempted;
          MPI_Reduce(&sbufi, &rbufi, 1, MPI_INT, MPI_SUM, 0, roots);
          n_swaps_attempted = rbufi / 2; // Divide by two b/c both from pairs added
        }

        /*-------------------------------------------------------------------------*/
        // 8. Update lookup tables
        if (this_local_proc == 0){
          MPI_Allgather(&i_replica_id, 1, MPI_INT, world2replicaid, 1, MPI_INT, roots);
          for (i=0; i<i_ncomms; i++) replicaid2world[world2replicaid[i]] = i;
        }
        MPI_Bcast(replicaid2world, i_ncomms, MPI_INT, 0, subcomm);

        // Timer
        MPI_Barrier(MPI_COMM_WORLD);
        double LoopTime_end = MPI_Wtime();

        /*-------------------------------------------------------------------------*/
        // 9. Print status to screen
        if(this_global_proc == 0) {
          bigint *current_ptr = (bigint *)lammps_extract_global(lmp, "ntimestep");
          bigint timestep = *current_ptr;
          printf("---------------------------------\n");
          printf("Timestep %ld : Swap dimension %d Steps %d\n", 
                  (long)timestep, idim, this_replica->dim_nevery[idim]);
          printf("Time for this dimension run (seconds): %.4f \n", LoopTime_end - LoopTime_start);
          printf("Swaps attempted  : %2d  ", n_swaps_attempted);
          printf("Swaps successful : %2d  ", n_swaps_successful);
          acceptance_ratio = 0.0;
          if (n_swaps_attempted > 0) acceptance_ratio = ((double)n_swaps_successful) / ((double)n_swaps_attempted);
          printf("Acceptance ratio : %6.4f\n", acceptance_ratio);
          TotalLoopTime += LoopTime_end - LoopTime_start;
          n_total_swaps_attempted  += n_swaps_attempted;
          n_total_swaps_successful += n_swaps_successful;
          average_acceptance_ratio += acceptance_ratio;
          int s;
          for (s=0; s<i_ncomms; ++s) {
            printf("  Subcomm %d : id = %d \n", s, world2replicaid[s]);
          }
        }

        // Let everyone catch up here
        MPI_Barrier(MPI_COMM_WORLD);

        n_swap_runs_completed++; // update swap runs count
      } // close idim
    } // close while loop 

/*----------------------------------------------------------------------------------
 *  Some final stats printing 
 */

    if (this_global_proc == 0) {
      printf("\n");
      printf("---------------------------------------------------------------\n");
      printf("Total wall time for COORDX:     %.4f seconds\n", TotalLoopTime);
      printf("Mean wall time b/w swaps:       %.4f seconds\n", TotalLoopTime / (double)(i_nswaps));
      printf("Mean acceptance ratio:          %.3f \n", average_acceptance_ratio / (double)(i_nswaps));
      printf("Total swaps attempted:          %d\n", n_total_swaps_attempted);
      printf("Total swaps successful:         %d\n", n_total_swaps_successful);
      printf("Total acceptance ratio:         %.3f\n", ((double)n_total_swaps_successful) / ((double)n_total_swaps_attempted) );
      printf("---------------------------------------------------------------\n");
    }

/*----------------------------------------------------------------------------------
 * clean it up
 */

    if (this_global_proc == 0) {
      printf("\nRun completed! Cleaning up LAMMPS...\n");
    }

    lammps_mod_inst(lmp, 3, NULL, "cleanup", NULL);
    lammps_mod_inst(lmp, 0, NULL, "finish", NULL);
    lammps_mod_inst(lmp, 4, NULL, "cleanup", NULL);

    free(world2root);
    free(replicaid2temp);
    free(world2replicaid);
    free(replicaid2world);
    free(world2temp);
    free(my_coords);
    free(partner_coords);
    free(my_velocs);
    free(partner_velocs);

}
