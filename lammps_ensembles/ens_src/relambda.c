/*
 * plumed.c
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

#include "replica.h"
#include "string.h"

/*----------------------------------------------------------------------------------
 * AWGL : Replica exchange with lamda scalar for off-diagonal coupling b/w MS-EVB states 
 */

void relambda(void *lmp, MPI_Comm subcomm, char* LID, int nsteps, int nevery, int ncomms, int comm, double temp, 
              char* fix, int seed, double lambda) 
{

/*----------------------------------------------------------------------------------
 * MPI things
 */
    
    MPI_Barrier(MPI_COMM_WORLD);
    // who i am in subcomm, how many in subcomm
    int this_proc, n_procs;
    MPI_Comm_rank(subcomm, &this_proc);
    MPI_Comm_size(subcomm, &n_procs);

    // who i am in COMM_WORLD
    int this_global_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &this_global_proc);

    // Total timer
    MPI_Barrier(MPI_COMM_WORLD);
    double TotalTime_start = MPI_Wtime();
    double TotalLoopTime = 0.0;

/*----------------------------------------------------------------------------------
 * grab info from args
 */

    int i_nsteps = nsteps;          	// number of (regular) steps to run
    int i_nevery = nevery;          	// frequency of swap
    int i_comm = comm;              	// instance ID
    int i_ncomms = ncomms;          	// number of instances
    int i_temp_id = i_comm;		// temp ID
    double i_temp = temp;               // temperature
    double i_lambda = lambda;           // lambda scalar
    int p;

/*----------------------------------------------------------------------------------
 * determine number of swaps and check that they divide evenly into run length
 */

    int i_nswaps = i_nsteps / i_nevery;
    if (i_nswaps * i_nevery != i_nsteps) {
      printf("Number of regular steps must divide number of swaps.\n");
      exit(1);
    }
    if (this_global_proc == 0)
      printf("Will run for %d swap steps.\n", i_nswaps);

/*----------------------------------------------------------------------------------
 * setup update and initialize lammps
 */
 
    lammps_mod_inst(lmp, 4, NULL, "setup", &i_nsteps);
    lammps_mod_inst(lmp, 0, NULL, "init", NULL);

/*----------------------------------------------------------------------------------
 * Turn on the lambda flag and set lambda
 */
    lammps_modify_EVB_data(lmp, fix, 1, NULL);
    lammps_modify_EVB_data(lmp, fix, 2, &i_lambda);


/*----------------------------------------------------------------------------------
 * grab boltzmann constant from lammps - depends on user "units" command
 */

    double *boltz_ptr = (double *)lammps_extract_global(lmp, "boltz");
    double boltz = *boltz_ptr;
#ifdef RELAMBDA_DEBUG
    if (this_global_proc == 0) printf("boltz = %f\n", boltz);
#endif
	
/*----------------------------------------------------------------------------------
 * initialize compute instance for potential energy calculation
 */
 
    lammps_mod_inst(lmp, 2, "thermo_pe", "addstep", &i_nevery);

/*----------------------------------------------------------------------------------
 * create communicator for roots on subcomms
 */

    int split_key;                                          // color argument
    MPI_Comm roots;                                         // newcomm argument
    if(this_proc == 0) split_key = 0;                       // grab roots on subcomm
    else split_key = 1;
    MPI_Comm_split(MPI_COMM_WORLD, split_key, 0, &roots);

/*----------------------------------------------------------------------------------
 * set up rngs
 */

    int i_seed = seed;
    int ranswapflag = 1;
    if(i_seed == 0) ranswapflag = 0; 
#ifdef RELAMBDA_DEBUG
    if(this_global_proc == 0) printf("ranswapflag = %d\n", ranswapflag);
#endif

    // Set up random number
    if (i_seed < 0) {
      // If it's negative, the global root decides from the current time, 
      // broadcasts, and other ranks add their global rank. All seeds should be unique.
      int time_seed;
      if (this_global_proc == 0) 
        time_seed = rng2_get_time_seed();
      MPI_Bcast(&time_seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
      rng2_seed(time_seed + this_global_proc);
#ifdef RELAMBDA_DEBUG
      if(this_global_proc == 0) printf("Time seed = %d\n", time_seed);
#endif
    }
    else {
      rng2_seed(i_seed + this_global_proc);
    }


/*----------------------------------------------------------------------------------
 * create lookup tables for useful values
 */

    // world2root - return global proc of root proc on a given subcomm
    int* world2root = (int *)malloc(sizeof(int) * i_ncomms);
    memset(world2root, 0, sizeof(int) * i_ncomms);
    if (this_proc == 0) {
      MPI_Allgather(&this_global_proc, 1, MPI_INT, world2root, 1, MPI_INT, roots);
    }
    MPI_Bcast(world2root, i_ncomms, MPI_INT, 0, subcomm);
	
    // world2tempid - return temp_id for a given comm world
    int* world2tempid = (int *)malloc(sizeof(int) * i_ncomms);
    memset(world2tempid, 0, sizeof(int) * i_ncomms);
    if (this_proc == 0) {
      MPI_Allgather(&i_temp_id, 1, MPI_INT, world2tempid, 1, MPI_INT, roots);
    }

    // Temporary LID
    char temp_LID[50];
    char my_LID[50];
    memcpy(my_LID, LID, 50*sizeof(char));
		
/*----------------------------------------------------------------------------------
 * initialize integrator
 */

    lammps_mod_inst(lmp, 3, NULL, "setup", NULL);

/*----------------------------------------------------------------------------------
 * print status header
 */

    if(this_global_proc == 0) {
        printf("----------------------------------------------------------------------------------\n");
	printf("Swap\t");
	for(p = 0; p < i_ncomms; p++) printf("L%d\t", p);
        printf(" Loop Time");
	printf("\n");
    }	

/*----------------------------------------------------------------------------------
 * set up for main loop 
 */

    int iswap, swap, dir, p_temp_id, partner_proc, partner_comm, lower_proc;
    double partner_energy, partner_lambda;
    MPI_Status status;
    double beta = 1.0 / (boltz * i_temp);

    int timestep = 0;
    if (this_proc == 0) {
      int *current_ptr = (int *)lammps_extract_global(lmp, "ntimestep");
      timestep = *current_ptr;
    }
    MPI_Bcast(&timestep, 1, MPI_INT, 0, subcomm);

    // Print initial status to screen
    if(this_proc == 0 && i_comm == 0) {
      printf("0\t"); 
      for(p = 0; p < i_ncomms; p++) {
        printf("%d\t", world2tempid[p]);
      }
      printf(" 0.0000\n"); 
    }

/*----------------------------------------------------------------------------------
 * main loop 
 */

    for (iswap = 0; iswap < i_nswaps; iswap += 1) {

        // Timer
        MPI_Barrier(MPI_COMM_WORLD);
        double LoopTime_start = MPI_Wtime();

        // 1. determine which direction matching takes place, global root decides
        if (this_global_proc == 0) {
          if (ranswapflag == 0)  dir = iswap % 2;
          else if (rng2() < 0.5) dir = 0;
          else                   dir = 1;
        }
        MPI_Bcast(&dir, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 2. match partner windows
        if (i_comm % 2 == dir) partner_comm = i_comm + 1;
        else                   partner_comm = i_comm - 1;
        // Test for only two replicas, they always swap
        if (i_ncomms == 2) {
          if (i_comm == 0) partner_comm = 1; 
          if (i_comm == 1) partner_comm = 0;
        }

        // 3. get partner global rank
        partner_proc = -1;
        lower_proc = 0;
        if (this_proc == 0) {
          i_temp_id = world2tempid[i_comm];
          if (partner_comm >= 0 && partner_comm < i_ncomms) {
            p_temp_id = world2tempid[partner_comm];
            partner_proc = world2root[partner_comm];
            if(this_global_proc < partner_proc) {
              lower_proc = 1;
            }
          } else {
            p_temp_id = -1;
          }
        }
        MPI_Bcast(&partner_proc, 1, MPI_INT, 0, subcomm);
        MPI_Bcast(&lower_proc, 1, MPI_INT, 0, subcomm);

        // *** MPI_COMM_WORLD is NOT synced *** //
        // 4. run for one period of timesteps (regular run)
        lammps_mod_inst(lmp, 3, NULL, "run", &i_nevery);
        timestep += i_nevery;

        // Just in case...
        MPI_Barrier(MPI_COMM_WORLD);
        // *** MPI_COMM_WORLD is now synced *** //

        // 5. get my current energy and lambda 
        double my_energy = lammps_extract_EVB_data(lmp, fix, 2, 0);
        double my_lambda = lammps_extract_EVB_data(lmp, fix, 0, 0);

        // 5.1 Add more steps to the queue
        lammps_mod_inst(lmp, 2, "thermo_pe", "addstep", &i_nevery);

#ifdef RELAMBDA_DEBUG
        if (this_proc == 0)
          printf("%d : my_lambda = %f\n", i_comm, my_lambda);
#endif
  
        // 6. figure out if swap is okay
        double buffer[2];
        swap = 0;
        if (partner_proc != -1) { // if it was left as default -1, skip swap attempt (relevant only to edge replicas 0 and N-1)
            if (this_proc == 0) {
                if (lower_proc) {
                    // lower proc recieves information...
		    MPI_Recv(buffer, 2, MPI_DOUBLE, partner_proc, 0, MPI_COMM_WORLD, &status);
                    partner_energy = buffer[0];
                    partner_lambda = buffer[1];
                    // then sends information to upper proc
                    buffer[0] = my_energy;
                    buffer[1] = my_lambda;
                    MPI_Send(buffer, 2, MPI_DOUBLE, partner_proc, 1, MPI_COMM_WORLD);
                    // recieve temp_LID
                    MPI_Recv(temp_LID, 50, MPI_CHAR, partner_proc, 4, MPI_COMM_WORLD, &status);
                    // send my_LID
                    MPI_Send(my_LID, 50, MPI_CHAR, partner_proc, 5, MPI_COMM_WORLD);
                } else {
                    // higher proc sends bias information to lower proc...
                    buffer[0] = my_energy;
                    buffer[1] = my_lambda;
                    MPI_Send(buffer, 2, MPI_DOUBLE, partner_proc, 0, MPI_COMM_WORLD);
                    // then recieves bias information from lower proc
		    MPI_Recv(buffer, 2, MPI_DOUBLE, partner_proc, 1, MPI_COMM_WORLD, &status);
                    partner_energy = buffer[0];
                    partner_lambda = buffer[1];
                    // send my_LID
                    MPI_Send(my_LID, 50, MPI_CHAR, partner_proc, 4, MPI_COMM_WORLD);
                    // recieve temp_LID
                    MPI_Recv(temp_LID, 50, MPI_CHAR, partner_proc, 5, MPI_COMM_WORLD, &status);
                }
            }
            // ** Root on subcomm needs to broadcast the new bias information to other procs in the subcomm ** // 
            // Broadcast LID stuff too
            MPI_Bcast(&partner_energy, 1, MPI_DOUBLE, 0, subcomm);
            MPI_Bcast(&partner_lambda, 1, MPI_DOUBLE, 0, subcomm);
            MPI_Bcast(temp_LID, 50, MPI_CHAR, 0, subcomm);

            double V_mi, V_mj, V_ni, V_nj;
            V_mi = V_mj = V_ni = V_nj = 0.0;

            if (lower_proc) {
	      // lower proc does the m calculations
              V_mi = my_energy; 

              // compute energy with partner's lambda
              lammps_modify_EVB_data(lmp, fix, 4, NULL); // turn off writing to evb.out
              lammps_modify_EVB_data(lmp, fix, 2, &partner_lambda);
              V_mj = lammps_extract_EVB_data(lmp, fix, 1, 0);

              // Recieve info from upper proc
              if (this_proc == 0) MPI_Recv(buffer, 2, MPI_DOUBLE, partner_proc, 2, MPI_COMM_WORLD, &status);
              // Send buffer to subcomm
              MPI_Bcast(buffer, 2, MPI_DOUBLE, 0, subcomm);
              V_nj = buffer[0];
              V_ni = buffer[1];
              // restore my lambda 
              lammps_modify_EVB_data(lmp, fix, 2, &my_lambda);
            } 
            else {
              // upper proc does the n calculations
              V_nj = my_energy; 

              // compute energy with partner's lambda
              lammps_modify_EVB_data(lmp, fix, 4, NULL); // turn off writing to evb.out
              lammps_modify_EVB_data(lmp, fix, 2, &partner_lambda);
              V_ni = lammps_extract_EVB_data(lmp, fix, 1, 0);

              // Send info to lower proc
              buffer[0] = V_nj;
              buffer[1] = V_ni;
              if (this_proc == 0) MPI_Send(buffer, 2, MPI_DOUBLE, partner_proc, 2, MPI_COMM_WORLD);
              // restore my lambda 
              lammps_modify_EVB_data(lmp, fix, 2, &my_lambda);
            }

            if (this_proc == 0) {
              if (lower_proc) {
                  // lower proc does the delta computation
                  double delta = (V_mj + V_ni) - (V_mi + V_nj);
                  double my_rand = rng2();
                  // make decision monte carlo style
                  if (delta <= 0.0) swap = 1; // criterion of e^0 or greater -> probability of 1
                  else if (my_rand < exp(-beta * delta)) swap = 1; 
#ifdef RELAMBDA_DEBUG
                  double prob = 1.0;
                  if (delta > 0.0) prob = exp(-beta * delta);
	          printf("comm %d : V_mj = %f V_ni = %f V_mi = %f V_nj = %f delta: %lf, probability: %lf, rand: %lf, swap: %d\n", 
                          i_comm, V_mj, V_ni, V_mi, V_nj, delta, prob, my_rand, swap);
#endif
                  // send decision to higher proc
                  MPI_Send(&swap, 1, MPI_INT, partner_proc, 3, MPI_COMM_WORLD);
              }
              else {
                  MPI_Recv(&swap, 1, MPI_INT, partner_proc, 3, MPI_COMM_WORLD, &status);
              }
            }
            // broadcast decision to subcomm
            MPI_Bcast(&swap, 1, MPI_INT, 0, subcomm);
        }
				
        // 7. perform swap
        // I currently am set to my partner's lambda from attempted swap above. 
        // So, only need to restore my lambda if not swapping. ALSO, must recompute energy and force!
        if (swap == 1) {
            // Set to partner's lambda 
            lammps_modify_EVB_data(lmp, fix, 2, &partner_lambda);
            // reset temp_id
            i_temp_id = p_temp_id;
            // reset LID
            memcpy(my_LID, temp_LID, 50*sizeof(char));
        }
        else {
            // 7.1 Recompute forces in case of altered lambda, no writing to evb.out though 
            lammps_extract_EVB_data(lmp, fix, 1, 0);
        }
        lammps_modify_EVB_data(lmp, fix, 1, NULL); // turn back on writing to evb.out

        // 8. Update lookup table
        if (this_proc == 0) {
          MPI_Allgather(&i_temp_id, 1, MPI_INT, world2tempid, 1, MPI_INT, roots);
        }

        // Timer
        MPI_Barrier(MPI_COMM_WORLD);
        double LoopTime_end = MPI_Wtime();

        // Print status to screen
        if (this_proc == 0 && i_comm == 0) {
          printf("%d\t", iswap+1); 
	  for (p = 0; p < i_ncomms; ++p) {
	    printf("%d\t", world2tempid[p]);
	  }
          printf(" %.4f", LoopTime_end - LoopTime_start); // wall time for loop
	  printf("\n");
          TotalLoopTime += LoopTime_end - LoopTime_start;
	}
    } // close iswap main loop

/*----------------------------------------------------------------------------------
 * Timer 
 */

    if(this_global_proc == 0)
      printf("----------------------------------------------------------------------------------\n");
    // Total timer
    MPI_Barrier(MPI_COMM_WORLD);
    double TotalTime_end = MPI_Wtime();
    if(this_global_proc == 0) {
      printf("Total wall time for loop:           %.4f seconds\n", TotalLoopTime); 
      printf("Total wall time for other RELambda: %.4f seconds\n", (TotalTime_end - TotalTime_start) - TotalLoopTime); 
      printf("Total wall time for RELambda:       %.4f seconds\n", TotalTime_end - TotalTime_start); 
      printf("----------------------------------------------------------------------------------\n");
    }

/*----------------------------------------------------------------------------------
 * Write final restart file, always 
 */

    if(this_global_proc == 0)
      printf("Writing final restart files.\n");
    MPI_Bcast(world2tempid, i_ncomms, MPI_INT, 0, subcomm);
    char *filename[1];
    char tmp[256]; 
    sprintf(tmp, "restart_final.%s", my_LID);
    filename[0] = (char*)tmp;
    lammps_write_restart(lmp, filename, timestep);

/*----------------------------------------------------------------------------------
 * clean it up
 */

    if(this_global_proc == 0) printf("RELambda has completed. Cleaning up memory and exiting...\n");

    lammps_mod_inst(lmp, 3, NULL, "cleanup", NULL);
    lammps_mod_inst(lmp, 0, NULL, "finish", NULL);
    lammps_mod_inst(lmp, 4, NULL, "cleanup", NULL);

    // ** Free the world maps ** // 
    free(world2root);
    free(world2tempid);
}


