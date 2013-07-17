/*
 * mreus.c 
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

#define COORD_CART      0
#define COORD_SPHERICAL 1
#define COORD_CYLINDER  2
#define COORD_PT        3

#define MAX_STEPS       10000000


/*----------------------------------------------------------------------------------
 * AWGL : Multidimensional REUS using Yuxing's umbrella fix in LAMMPS 
 *        Incudes exchanges for lambda and/or temperature
 */

void mreus(void *lmp, MPI_Comm subcomm, int ncomms, int comm,  
           char *EVBfix, char *fix, int seed, int coordtype, int nsteps_short, int dump, int dump_swap, 
           Replica *this_replica) {

/*----------------------------------------------------------------------------------
 * MPI things
 */
    
    MPI_Barrier(MPI_COMM_WORLD);
    // who i am in subcomm, how many in subcomm
    int this_local_proc, n_procs;
    MPI_Comm_rank(subcomm, &this_local_proc);
    MPI_Comm_size(subcomm, &n_procs);

    // who i am in COMM_WORLD
    int this_global_proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &this_global_proc);

    // Total timer
    MPI_Barrier(MPI_COMM_WORLD);
    double TotalTime_start = MPI_Wtime();

/*----------------------------------------------------------------------------------
 * grab info from args
 */

    int i;                              // just a generic loop index
    int i_comm = comm;              	// instance ID
    int i_ncomms = ncomms;          	// number of instances
    this_replica->comm = i_comm;        // instance ID for this replica

    int i_replica_id   = this_replica->id;            // replica ID
    double i_lambda    = this_replica->lambda;        // initial lambda scalar
    int i_ndimensions  = this_replica->N_dimensions;  // number of dimensions
    int i_reus_dim     = this_replica->reus_dim;      // which dimension, if any, is the REUS swapping dimension
    int i_temp_dim     = this_replica->temp_dim;      // which dimension, if any, is the temperature swapping dimension
    int i_lambda_dim   = this_replica->lambda_dim;    // which dimension, if any, is the lambda swapping dimension

    int i_nsteps = 0;                                 // total number of steps to run
    int i_nswaps = 0;                                 // total number of swaps for entire run
    int i_nevery = 0;                                 // frequency of swap, will depend on dimension
    int i_nsteps_short = nsteps_short;                // number of (short) steps to run
    for (i=0; i<i_ndimensions; ++i) {
      i_nsteps += this_replica->dim_run[i];
      i_nswaps += this_replica->dim_run[i] / this_replica->dim_nevery[i];
    }
    if (this_global_proc == 0) {
      printf("Total MD steps to take:  %d\n", i_nsteps);
      printf("Total swap runs to make: %d\n", i_nswaps);
    }
    int i_dump  = dump;                               // dump to COLVAR.[CVID] frequency 

    int i_temp_id = i_comm;		// temp ID
    int toggle = 1;                     // filename toggling for restart files
    int p;

    char CVID[50]; // Collecvtive Variable ID, set to i_replica_id
    sprintf(CVID, "%d", i_replica_id);

/*----------------------------------------------------------------------------------
 * Check for asynchronous load balancing 
 */
    int async = 0;
    if (nsteps_short > 0) async = 1;
    if (async && this_global_proc == 0)
      printf("Asynchronous load balance turned on with short runs of %d steps.\n", nsteps_short); 

/*----------------------------------------------------------------------------------
 * setup update and initialize lammps
 */
 
    int i_max_steps = MAX_STEPS; // To avoid short runs overstepping fixed number of steps 
    lammps_mod_inst(lmp, 4, NULL, "setup", &i_max_steps);
    //lammps_mod_inst(lmp, 4, NULL, "setup", &i_nsteps);
    lammps_mod_inst(lmp, 0, NULL, "init", NULL);

/*----------------------------------------------------------------------------------
 * Turn on the lambda flag and set lambda, if necessary
 */
   if (i_lambda_dim >= 0) {
      lammps_modify_EVB_data(lmp, EVBfix, 1, NULL);
      lammps_modify_EVB_data(lmp, EVBfix, 2, &i_lambda);
   }


/*----------------------------------------------------------------------------------
 * grab boltzmann constant from lammps - depends on user "units" command
 */

    double *boltz_ptr = (double *)lammps_extract_global(lmp, "boltz");
    double boltz = *boltz_ptr;
#ifdef MREUS_DEBUG
    if (this_global_proc == 0) printf("boltz = %f\n", boltz);
#endif
	
/*----------------------------------------------------------------------------------
 * initialize compute instance for potential energy calculation
 */
 
    //lammps_mod_inst(lmp, 2, "thermo_pe", "addstep", &i_nevery);

/*----------------------------------------------------------------------------------
 * create communicator for roots on subcomms
 */

    int split_key;                                          // color argument
    MPI_Comm roots;                                         // newcomm argument
    if(this_local_proc == 0) split_key = 0;                       // grab roots on subcomm
    else split_key = 1;
    MPI_Comm_split(MPI_COMM_WORLD, split_key, 0, &roots);

/*----------------------------------------------------------------------------------
 * set up rngs
 */

    int i_seed = seed;
    int ranswapflag = 1;
    if(i_seed == 0) ranswapflag = 0; 
#ifdef MREUS_DEBUG
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
#ifdef MREUS_DEBUG
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
    if (this_local_proc == 0) {
      MPI_Allgather(&this_global_proc, 1, MPI_INT, world2root, 1, MPI_INT, roots);
    }
    MPI_Bcast(world2root, i_ncomms, MPI_INT, 0, subcomm);

    // replicaid2temp - return given temperature for a given replica id
    double *replicaid2temp = (double *)malloc(sizeof(double) * i_ncomms);
    if(this_local_proc == 0)
        MPI_Allgather(&this_replica->temperature, 1, MPI_DOUBLE, replicaid2temp, 1, MPI_DOUBLE, roots);
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
	
    // world2tempid - return temp_id for a given comm world
    int* world2tempid = (int *)malloc(sizeof(int) * i_ncomms);
    memset(world2tempid, 0, sizeof(int) * i_ncomms);
    if (this_local_proc == 0) {
      MPI_Allgather(&i_temp_id, 1, MPI_INT, world2tempid, 1, MPI_INT, roots);
    }

    // world2temperature - return the temperature for a given world 
    double *world2temperature = (double *)malloc(sizeof(double) * i_ncomms);
    if (this_local_proc == 0) {
      MPI_Allgather(&this_replica->temperature, 1, MPI_DOUBLE, world2temperature, 1, MPI_DOUBLE, roots);
    }
    MPI_Bcast(world2temperature, i_ncomms, MPI_DOUBLE, 0, subcomm);
#ifdef MREUS_DEBUG
    if (this_global_proc == 0) {
      printf("Initial temperature configuration:\n");
      for (i=0; i<i_ncomms; i++) {
        printf("%d : %f\n", i, world2temperature[i]);
      }
    }
#endif

    // Temporary CVID
    char temp_CVID[50];
    char my_CVID[50];
    memcpy(my_CVID, CVID, 50*sizeof(char));
		
/*----------------------------------------------------------------------------------
 * initialize integrator
 */

    lammps_mod_inst(lmp, 3, NULL, "setup", NULL);

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
 * set up for main loop 
 */

    int iswap, swap, dir, p_temp_id, partner_proc, partner_comm, lower_proc;
    double bias_v;                                // my current bias potential 
    double bias_dx[3], bias_partner_dx[3];        // current position along collective variable
    double bias_ref[3], bias_partner_ref[3];      // equilibrium position for collective variable (this is what gets swapped!)
    double bias_kappa[3], bias_partner_kappa[3];  // force constant (this also gets swapped)!
    double bias_xa0[3], bias_partner_xa0[3];      // equilibrium position for CEC (this also gets swapped!) 
    double adjust;                                // possible periodicity adjustment for bias potential
    adjust = 0.0; // ignore for now
    double h_save;

    MPI_Status status;

    // For swapping output file names
    char *my_dumpfile      = (char *)malloc(sizeof(char) * MAXCHARS);
    char *partner_dumpfile = (char *)malloc(sizeof(char) * MAXCHARS);

    // Grab umbrella data
    get_umbrella_data(lmp, fix, bias_dx, bias_ref, bias_kappa, bias_xa0, &bias_v, &h_save, coordtype);

#ifdef MREUS_DEBUG
    if (i_comm == 0 && this_local_proc == 0) {
      printf("v = %f dx = %f %f %f ref = %f %f %f k = %f xa0 = %f %f %f\n", bias_v, bias_dx[0], bias_dx[1], bias_dx[2],
	      bias_ref[0], bias_ref[1], bias_ref[2], bias_kappa[0], bias_xa0[0], bias_xa0[1], bias_xa0[2]); 
    }
#endif

    // ** Before starting the simulation, set up the COLVAR.# file headers ** //
    bigint timestep = 0;
    bigint *current_ptr = (bigint *)lammps_extract_global(lmp, "ntimestep");
    timestep = *current_ptr;
    if (this_local_proc == 0) {
      printf("Initial time step on subcomm %d is %ld\n", i_comm, (long)timestep);
    }
    if (this_local_proc == 0) {
      write_to_colvar_init_vec(bias_kappa, bias_ref, bias_xa0, i_comm, my_CVID);
      write_to_colvar_vec(timestep, bias_dx, h_save, bias_v, i_temp_id, i_comm, my_CVID);
    }
    MPI_Barrier(MPI_COMM_WORLD);


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
 * main loops
 * Outer loop over total number of swaps
 * Inner loop over dimensions, go through each dimension sequentially in order 
 */

    // ** The outer loop ** //
    // Loop until all swap runs are done
    int n_swap_runs_completed = 0;
    while (n_swap_runs_completed < i_nswaps) {

      iswap = n_swap_runs_completed / i_ndimensions;

      // ** The inner loop over dimensions ** //
      int idim;
      for (idim=0; idim < i_ndimensions; ++idim) {

	// Timer
	MPI_Barrier(MPI_COMM_WORLD);
	double LoopTime_start = MPI_Wtime();
        // Set data for this dimension
        i_nevery = this_replica->dim_nevery[idim];
	swap = 0;
	n_swaps_attempted = 0;
	n_swaps_successful = 0;
        sprintf(CVID, "%d", i_replica_id);
        memcpy(my_CVID, CVID, 50*sizeof(char));
        double beta = 1.0 / (boltz * this_replica->temperature); 

#ifdef MREUS_DEBUG
        if (this_global_proc == 0) {
          printf("Now on dimension %d. n_swap_runs_completed = %d\n", idim, n_swap_runs_completed);
        }
#endif

	// 1. determine which direction matching takes place, global root decides
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
#ifdef MREUS_DEBUG
        if (this_global_proc == 0) printf("Swap dir = %d\n", dir);
#endif

	// 2. match partner windows
	partner_comm = -1;
	int dim_num = this_replica->dim_num[idim];
	if (dir == -1) {
	  if (dim_num % 2) partner_comm = this_replica->neighbors[2*idim+1];
	  else             partner_comm = this_replica->neighbors[2*idim  ];
	} else {
	  if (dim_num % 2) partner_comm = this_replica->neighbors[2*idim  ];
	  else             partner_comm = this_replica->neighbors[2*idim+1];
	}
	int p_replica_id = -1;
	if (partner_comm >= 0) p_replica_id = world2replicaid[partner_comm];

	// 3. get partner global rank
	partner_proc = -1;
	lower_proc = 0;
	if (this_local_proc == 0) {
	  i_temp_id = world2tempid[i_comm];
	  if (partner_comm >= 0 && partner_comm < i_ncomms) {
	    p_temp_id = world2tempid[partner_comm];
	    //printf("i_temp_id %d : %d-->%d : p_temp_id = %d\n", i_temp_id, i_comm, partner_comm, p_temp_id); //debugging
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

#ifdef MREUS_DEBUG 
	if(this_local_proc == 0) {
	  printf("subcomm %d: replica_id %d p_replica_id = %d partner_comm = %d partner_proc = %d dim_num %d\n",
		  i_comm, world2replicaid[i_comm], p_replica_id, partner_comm, partner_proc, dim_num);
	}
#endif

	//determine number of swaps and check that they divide evenly into run length
	int div_step  = i_nevery / i_dump;
	if (div_step < 1) {
	  if (this_global_proc == 0) {
	    printf("ERROR.\n");
	    printf("Swap frequency: %d \n", i_nevery);
	    printf("Dump frequency: %d \n", i_dump);
	    printf("swap/dump = %d/%d = %d < 1\n", i_nevery, i_dump, div_step);
	    printf("swap/dump must be equal to or greater than 1. Please adjust input.\n");
	  }
	  MPI_Barrier(MPI_COMM_WORLD);
	  exit(1);
	}

	// 4. Run for one period of timesteps (regular run)
        // **** All dimension types do the run here. And all dimensions are capable of using the asynch load balancer. **** //
	// *** MPI_COMM_WORLD is NOT synced *** //
	// Run until we hit a dump step again, then dump. This is just to make the steps line up with the dumping below.
	while (timestep % i_dump != 0) { 
	  lammps_mod_inst(lmp, 2, "thermo_pe", "addstep", &i_nsteps_short);
	  lammps_mod_inst(lmp, 3, NULL, "run", &i_nsteps_short); 
	  current_ptr = (bigint *)lammps_extract_global(lmp, "ntimestep");
	  timestep = *current_ptr;
	  if (timestep % i_dump == 0) {
	    get_umbrella_data(lmp, fix, bias_dx, bias_ref, bias_kappa, bias_xa0, &bias_v, &h_save, coordtype);
	    if (this_local_proc == 0) write_to_colvar_vec(timestep, bias_dx, h_save, bias_v, i_temp_id, i_comm, my_CVID);
	  }
	}
	// Loop over div_step runs till we have run for i_nevery. This is the "regular" run broken into pieces for dumps.
	int d;
	for (d=0; d<div_step; d++) {
	  lammps_mod_inst(lmp, 2, "thermo_pe", "addstep", &i_dump);
	  lammps_mod_inst(lmp, 3, NULL, "run", &i_dump);
	  current_ptr = (bigint *)lammps_extract_global(lmp, "ntimestep");
	  timestep = *current_ptr;
	  get_umbrella_data(lmp, fix, bias_dx, bias_ref, bias_kappa, bias_xa0, &bias_v, &h_save, coordtype);
	  if (this_local_proc == 0) write_to_colvar_vec(timestep, bias_dx, h_save, bias_v, i_temp_id, i_comm, my_CVID);
	} 


	if (i_comm != 0 && async) {
	  // ***** Non-comm 0 guys ***** //
	  int my_regular_run_finished = 1;
	  int all_regular_run_finished = 0;
	  int test_flag = 0;

	  // 4.a Inform the root comm that I finished my regular run
	  MPI_Request my_fin_req;
	  if (this_local_proc == 0) MPI_Isend(&my_regular_run_finished, 1, MPI_INT, 0, 1, roots, &my_fin_req);
	  // 4.b Post recieve message from root comm on completion status of everyone else
	  MPI_Request all_fin_req;
	  if (this_local_proc == 0) MPI_Irecv(&all_regular_run_finished, 1, MPI_INT, 0, 2, roots, &all_fin_req);
	  // 4.c Has everyone finished regular run?... 
	  MPI_Status status1;
	  if (this_local_proc == 0) MPI_Test(&all_fin_req, &test_flag, &status1);
	  MPI_Bcast(&test_flag, 1, MPI_INT, 0, subcomm); // Inform my subcomm about test
	  // 4.c If everyone has not finished the regular run, then do a short asynchronous run until everyone has 
	  while (!test_flag) {
	    
	    lammps_mod_inst(lmp, 2, "thermo_pe", "addstep", &i_nsteps_short);
	    lammps_mod_inst(lmp, 3, NULL, "run", &i_nsteps_short);
	    current_ptr = (bigint *)lammps_extract_global(lmp, "ntimestep");
	    timestep = *current_ptr;
	    if (timestep % i_dump == 0) {
	      get_umbrella_data(lmp, fix, bias_dx, bias_ref, bias_kappa, bias_xa0, &bias_v, &h_save, coordtype);
	      if (this_local_proc == 0) write_to_colvar_vec(timestep, bias_dx, h_save, bias_v, i_temp_id, i_comm, my_CVID);
	    }
	    if (this_local_proc == 0) MPI_Test(&all_fin_req, &test_flag, &status1);
	    MPI_Bcast(&test_flag, 1, MPI_INT, 0, subcomm); // Inform my subcomm about test

	  } 
  
	} else if (i_comm == 0 && async) {
	  // ***** Comm 0 master of completions ***** //

	  // Create recieve list
	  MPI_Request* req_list = (MPI_Request*)malloc(sizeof(MPI_Request) * i_ncomms);
	  int* run_list = (int*)malloc(sizeof(int) * i_ncomms); 
	  run_list[0] = 1;
	  for (i=1; i<i_ncomms; ++i) run_list[i] = 0;

	  int i_have_recvd_them_all = 0;
	  if (this_local_proc == 0) {
	    // Post recieves from all the other comms about regular run completion
	    for (i=1; i<i_ncomms; ++i) { 
	      int tmp_num = 0;
	      MPI_Irecv(&tmp_num, 1, MPI_INT, i, 1, roots, &req_list[i]);
	    }
	    // Test the recieves
	    for (i=1; i<i_ncomms; ++i) {
	      MPI_Status status;
	      int flag = 0;
	      MPI_Test(&req_list[i], &flag, &status);
	      if (flag) run_list[i] = 1;
	    }
	    // Check sum of recieves
	    int sum = 1;
	    for (i=1; i<i_ncomms; ++i) sum += run_list[i];
	    if (sum == i_ncomms) i_have_recvd_them_all = 1;
	    
#ifdef MREUS_DEBUG
	    for (i=0; i<i_ncomms; ++i) printf("run_list[%d] = %d\n", i, run_list[i]);
#endif
	  }
	  MPI_Bcast(&i_have_recvd_them_all, 1, MPI_INT, 0, subcomm); // Inform the other procs in my subcomm

	  // Until we get them all, keep doing short runs
	  while (!i_have_recvd_them_all) {

	    lammps_mod_inst(lmp, 2, "thermo_pe", "addstep", &i_nsteps_short);
	    lammps_mod_inst(lmp, 3, NULL, "run", &i_nsteps_short);
	    current_ptr = (bigint *)lammps_extract_global(lmp, "ntimestep");
	    timestep = *current_ptr;
	    if (timestep % i_dump == 0) {
	      get_umbrella_data(lmp, fix, bias_dx, bias_ref, bias_kappa, bias_xa0, &bias_v, &h_save, coordtype);
	      if(this_local_proc == 0) write_to_colvar_vec(timestep, bias_dx, h_save, bias_v, i_temp_id, i_comm, my_CVID);
	    }

	    if (this_local_proc == 0) {
	      // Test the recieves
	      for (i=1; i<i_ncomms; ++i) {
		if (run_list[i] == 0) {
		  MPI_Status status;
		  int flag = 0;
		  MPI_Test(&req_list[i], &flag, &status);
		  if (flag) run_list[i] = 1;
		}
	      }
	      // Check sum of recieves
	      int sum = 1;
	      for (i=1; i<i_ncomms; ++i) sum += run_list[i];
	      if (sum == i_ncomms) i_have_recvd_them_all = 1;

#ifdef MREUS_DEBUG
	      for (i=0; i<i_ncomms; ++i) printf("run_list[%d] = %d\n", i, run_list[i]);
#endif
	    }
	    MPI_Bcast(&i_have_recvd_them_all, 1, MPI_INT, 0, subcomm); // Inform the other procs in my subcomm
	  } 

	  if (this_local_proc == 0) {
	    // Post blocking sends to all other roots to let them know everyone has finished the regular run
	    int tmp_num = 1;
	    for (i=1; i<i_ncomms; ++i) MPI_Send(&tmp_num, 1, MPI_INT, i, 2, roots);
#ifdef MREUS_DEBUG
	    printf("All comms have completed their regular run.\n");
#endif
	  }
	  free(req_list);
	  free(run_list);
	}

	// Just in case...
	MPI_Barrier(MPI_COMM_WORLD);
	// *** MPI_COMM_WORLD is now synced *** //


        // ************ REUS dimension ************* //
        if (idim == i_reus_dim) {

	    // 5. extract current bias information of this instance 
	    get_umbrella_data(lmp, fix, bias_dx, bias_ref, bias_kappa, bias_xa0, &bias_v, &h_save, coordtype);
      
	    // 6. figure out if swap is okay
	    double buffer[12];
	    if (partner_proc != -1) { // if it was left as default -1, skip swap attempt (relevant only to edge replicas 0 and N-1)
		if (this_local_proc == 0) {
		    n_swaps_attempted = 1;
		    if (lower_proc) {
			// lower proc recieves bias information...
			MPI_Recv(buffer, 12, MPI_DOUBLE, partner_proc, 0, MPI_COMM_WORLD, &status);
			bias_partner_dx[0]    = buffer[0];
			bias_partner_dx[1]    = buffer[1];
			bias_partner_dx[2]    = buffer[2];
			bias_partner_ref[0]   = buffer[3];
			bias_partner_ref[1]   = buffer[4];
			bias_partner_ref[2]   = buffer[5];
			bias_partner_kappa[0] = buffer[6];
			bias_partner_kappa[1] = buffer[7];
			bias_partner_kappa[2] = buffer[8];
			bias_partner_xa0[0]   = buffer[9];
			bias_partner_xa0[1]   = buffer[10];
			bias_partner_xa0[2]   = buffer[11];
			// then sends bias information to upper proc
			buffer[0]  = bias_dx[0];
			buffer[1]  = bias_dx[1];
			buffer[2]  = bias_dx[2];
			buffer[3]  = bias_ref[0];
			buffer[4]  = bias_ref[1];
			buffer[5]  = bias_ref[2];
			buffer[6]  = bias_kappa[0];
			buffer[7]  = bias_kappa[1];
			buffer[8]  = bias_kappa[2];
			buffer[9]  = bias_xa0[0];
			buffer[10] = bias_xa0[1];
			buffer[11] = bias_xa0[2];
			MPI_Send(buffer, 12, MPI_DOUBLE, partner_proc, 1, MPI_COMM_WORLD);
			// recieve temp_CVID
			MPI_Recv(temp_CVID, 50, MPI_CHAR, partner_proc, 4, MPI_COMM_WORLD, &status);
			// send my_CVID
			MPI_Send(my_CVID, 50, MPI_CHAR, partner_proc, 5, MPI_COMM_WORLD);
		    } else {
			// higher proc sends bias information to lower proc...
			buffer[0]  = bias_dx[0];
			buffer[1]  = bias_dx[1];
			buffer[2]  = bias_dx[2];
			buffer[3]  = bias_ref[0];
			buffer[4]  = bias_ref[1];
			buffer[5]  = bias_ref[2];
			buffer[6]  = bias_kappa[0];
			buffer[7]  = bias_kappa[1];
			buffer[8]  = bias_kappa[2];
			buffer[9]  = bias_xa0[0];   
			buffer[10] = bias_xa0[1];   
			buffer[11] = bias_xa0[2];   
			MPI_Send(buffer, 12, MPI_DOUBLE, partner_proc, 0, MPI_COMM_WORLD);
			// then recieves bias information from lower proc
			MPI_Recv(buffer, 12, MPI_DOUBLE, partner_proc, 1, MPI_COMM_WORLD, &status);
			bias_partner_dx[0]    = buffer[0];
			bias_partner_dx[1]    = buffer[1];
			bias_partner_dx[2]    = buffer[2];
			bias_partner_ref[0]   = buffer[3];
			bias_partner_ref[1]   = buffer[4];
			bias_partner_ref[2]   = buffer[5];
			bias_partner_kappa[0] = buffer[6];
			bias_partner_kappa[1] = buffer[7];
			bias_partner_kappa[2] = buffer[8];
			bias_partner_xa0[0]   = buffer[9];
			bias_partner_xa0[1]   = buffer[10];
			bias_partner_xa0[2]   = buffer[11];
			// send my_CVID
			MPI_Send(my_CVID, 50, MPI_CHAR, partner_proc, 4, MPI_COMM_WORLD);
			// recieve temp_CVID
			MPI_Recv(temp_CVID, 50, MPI_CHAR, partner_proc, 5, MPI_COMM_WORLD, &status);
		    }
		}
		// ** Root on subcomm needs to broadcast the new bias information to other procs in the subcomm ** // 
		// Broadcast CVID stuff too
		MPI_Bcast(bias_partner_ref, 3, MPI_DOUBLE, 0, subcomm);
		MPI_Bcast(bias_partner_kappa, 3, MPI_DOUBLE, 0, subcomm);
		MPI_Bcast(bias_partner_xa0, 3, MPI_DOUBLE, 0, subcomm);
		MPI_Bcast(temp_CVID, 50, MPI_CHAR, 0, subcomm);

		double V_mi, V_mj, V_ni, V_nj;
		V_mi = V_mj = V_ni = V_nj = 0.0;

		if (lower_proc) {
		  // lower proc does the m calculations
		  V_mi = bias_v; // used from above 
		  // compute bias with partner's bias
		  lammps_compute_bias_stuff_for_external(lmp, fix, bias_partner_kappa, bias_partner_ref, bias_xa0);
		  V_mj = lammps_extract_umbrella_data(lmp, fix, -1, 0);
		  if (this_local_proc == 0) {
		    // Recieve info from upper proc
		    MPI_Recv(buffer, 2, MPI_DOUBLE, partner_proc, 2, MPI_COMM_WORLD, &status);
		  }
		  // Send buffer to subcomm
		  MPI_Bcast(buffer, 2, MPI_DOUBLE, 0, subcomm);
		  V_nj = buffer[0];
		  V_ni = buffer[1];
		} else {
		  // upper proc does the n calculations
		  V_nj = bias_v; // used from above                  
		  // compute bias with partner's bias
		  lammps_compute_bias_stuff_for_external(lmp, fix, bias_partner_kappa, bias_partner_ref, bias_xa0);
		  V_ni = lammps_extract_umbrella_data(lmp, fix, -1, 0); 
		  // Send info to lower proc
		  buffer[0] = V_nj;
		  buffer[1] = V_ni;
		  if (this_local_proc == 0) MPI_Send(buffer, 2, MPI_DOUBLE, partner_proc, 2, MPI_COMM_WORLD);
		}

		if (this_local_proc == 0) {
		  if(lower_proc) {
		      // lower proc does the delta computation
		      double delta = (V_mj + V_ni) - (V_mi + V_nj);
		      double my_rand = rng2();
		      // make decision monte carlo style
		      if(delta <= 0.0) swap = 1; // criterion of e^0 or greater -> probability of 1
		      else if(my_rand < exp(-beta * delta)) swap = 1; 
#ifdef MREUS_DEBUG
		      double prob = 1.0;
		      if(delta > 0.0) prob = exp(-beta * delta);
		      printf("V_mj = %f V_ni = %f V_mi = %f V_nj = %f delta: %lf, probability: %lf, rand: %lf, swap: %d\n", 
			      V_mj, V_ni, V_mi, V_nj, delta, prob, my_rand, swap);
#endif
		      // send decision to higher proc
		      MPI_Send(&swap, 1, MPI_INT, partner_proc, 3, MPI_COMM_WORLD);
		  }
		  else {
		      MPI_Recv(&swap, 1, MPI_INT, partner_proc, 3, MPI_COMM_WORLD, &status);
		  }
		}
	    }
	    // broadcast decision to subcomm
	    MPI_Bcast(&swap, 1, MPI_INT, 0, subcomm);

	    // 7. perform swap
	    if(swap == 1) {

		// Modify which bias I now have. Change it to my partner's.
		lammps_modify_umbrella_data(lmp, fix, 1, bias_partner_ref);  
		lammps_modify_umbrella_data(lmp, fix, 2, bias_partner_kappa);  
		//lammps_modify_umbrella_data(lmp, fix, 4, bias_partner_xa0);  

#ifdef MREUS_DEBUG
		double after[3];
		after[0] = lammps_extract_umbrella_data(lmp, fix, 1, 0);
		after[1] = lammps_extract_umbrella_data(lmp, fix, 1, 1);
		after[2] = lammps_extract_umbrella_data(lmp, fix, 1, 2);
	        if (this_local_proc == 0) {
		  printf("Comm %d ref: before = %f %f %f after = %f %f %f\n", i_comm, 
		  	bias_ref[0], bias_ref[1], bias_ref[2], after[0], after[1], after[2]);
                }
#endif
		// reset CVID
		memcpy(my_CVID, temp_CVID, 50*sizeof(char));
	    }
	} 

        // ************ RELAMBDA dimension ************* //
        else if (idim == i_lambda_dim) {

          double partner_energy, partner_lambda;

          // 5. get my current energy and lambda 
          double my_energy = lammps_extract_EVB_data(lmp, EVBfix, 2, 0);
          double my_lambda = lammps_extract_EVB_data(lmp, EVBfix, 0, 0);
#ifdef MREUS_DEBUG
          if (this_local_proc == 0) printf("Comm %d : my_lambda = %f\n", i_comm, my_lambda);
#endif
          // 6. figure out if swap is okay
          double buffer[2];
          if (partner_proc != -1) { // if it was left as default -1, skip swap attempt (relevant only to edge replicas 0 and N-1)
              if (this_local_proc == 0) {
		  n_swaps_attempted = 1;
                  if (lower_proc) {
                      // lower proc recieves information...
                      MPI_Recv(buffer, 2, MPI_DOUBLE, partner_proc, 0, MPI_COMM_WORLD, &status);
                      partner_energy = buffer[0];
                      partner_lambda = buffer[1];
                      // then sends information to upper proc
                      buffer[0] = my_energy;
                      buffer[1] = my_lambda;
                      MPI_Send(buffer, 2, MPI_DOUBLE, partner_proc, 1, MPI_COMM_WORLD);
                  } else {
                      // higher proc sends bias information to lower proc...
                      buffer[0] = my_energy;
                      buffer[1] = my_lambda;
                      MPI_Send(buffer, 2, MPI_DOUBLE, partner_proc, 0, MPI_COMM_WORLD);
                      // then recieves bias information from lower proc
                      MPI_Recv(buffer, 2, MPI_DOUBLE, partner_proc, 1, MPI_COMM_WORLD, &status);
                      partner_energy = buffer[0];
                      partner_lambda = buffer[1];
                  }
              }
              // ** Root on subcomm needs to broadcast the new bias information to other procs in the subcomm ** // 
              // Broadcast LID stuff too
              MPI_Bcast(&partner_energy, 1, MPI_DOUBLE, 0, subcomm);
              MPI_Bcast(&partner_lambda, 1, MPI_DOUBLE, 0, subcomm);

              double V_mi, V_mj, V_ni, V_nj;
              V_mi = V_mj = V_ni = V_nj = 0.0;
  
              if (lower_proc) {
                // lower proc does the m calculations
                V_mi = my_energy;
  
                // compute energy with partner's lambda
                lammps_modify_EVB_data(lmp, EVBfix, 4, NULL); // turn off writing to evb.out
                lammps_modify_EVB_data(lmp, EVBfix, 2, &partner_lambda);
                V_mj = lammps_extract_EVB_data(lmp, EVBfix, 1, 0);
  
                // Recieve info from upper proc
                if (this_local_proc == 0) MPI_Recv(buffer, 2, MPI_DOUBLE, partner_proc, 2, MPI_COMM_WORLD, &status);
                // Send buffer to subcomm
                MPI_Bcast(buffer, 2, MPI_DOUBLE, 0, subcomm);
                V_nj = buffer[0];
                V_ni = buffer[1];
                // restore my lambda 
                lammps_modify_EVB_data(lmp, EVBfix, 2, &my_lambda);
              }
              else {
                // upper proc does the n calculations
                V_nj = my_energy;
  
                // compute energy with partner's lambda
                lammps_modify_EVB_data(lmp, EVBfix, 4, NULL); // turn off writing to evb.out
                lammps_modify_EVB_data(lmp, EVBfix, 2, &partner_lambda);
                V_ni = lammps_extract_EVB_data(lmp, EVBfix, 1, 0);
  
                // Send info to lower proc
                buffer[0] = V_nj;
                buffer[1] = V_ni;
                if (this_local_proc == 0) MPI_Send(buffer, 2, MPI_DOUBLE, partner_proc, 2, MPI_COMM_WORLD);
                // restore my lambda 
                lammps_modify_EVB_data(lmp, EVBfix, 2, &my_lambda);
              }

              if (this_local_proc == 0) {
                if (lower_proc) {
                    // lower proc does the delta computation
                    double delta = (V_mj + V_ni) - (V_mi + V_nj);
                    double my_rand = rng2();
                    // make decision monte carlo style
                    if (delta <= 0.0) swap = 1; // criterion of e^0 or greater -> probability of 1
                    else if (my_rand < exp(-beta * delta)) swap = 1;
#ifdef MREUS_DEBUG
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
              lammps_modify_EVB_data(lmp, EVBfix, 2, &partner_lambda);
          }
          else {
              // 7.1 Recompute forces in case of altered lambda, no writing to evb.out though 
              lammps_extract_EVB_data(lmp, EVBfix, 1, 0);
          }
          lammps_modify_EVB_data(lmp, EVBfix, 1, NULL); // turn back on writing to evb.out

        }

        // ************ TEMPER dimension ************* //
        else if (idim == i_temp_dim) {
          if (partner_proc != -1) {
            // Extract the potential energy for parallel tempering
            // Note: In case of REUS, the bias potential cancels in deltaE for TEMPER dimension, so only need potential energy
            double *pe_ptr = (double *)lammps_extract_compute(lmp, "thermo_pe", 0, 0);
            double pe = *pe_ptr;
            double pe_partner;

            if (this_local_proc == 0) {
                n_swaps_attempted = 1;
                if (this_global_proc > partner_proc) {
                  // higher proc sends pe to lower proc
                  MPI_Send(&pe, 1, MPI_DOUBLE, partner_proc, 0, MPI_COMM_WORLD);
                } else {
                  // lower proc recieves
                  MPI_Recv(&pe_partner, 1, MPI_DOUBLE, partner_proc, 0, MPI_COMM_WORLD, &status);
                }
                // lower proc does calculations
                if (this_global_proc < partner_proc) {
                    double delta = (pe_partner - pe) *
                                   (1.0 / (boltz * world2temperature[partner_comm]) -
                                    1.0 / (boltz * world2temperature[i_comm]));
                    double my_rand = rng2();
                    // make decision monte carlo style
                    if (delta >= 0.0) swap = 1; // criterion of e^0 or greater -> probability of 1
                    else if (my_rand < exp(delta)) swap = 1;
#ifdef MREUS_DEBUG
                    double prob = exp(delta);
                    if (delta >= 0.0) prob = 1.0;
                    printf("%d<-->%d pe: %lf p_pe: %lf  delta: %lf, probability: %f, rand: %lf swap: %d\n",
                            i_comm, partner_comm, pe, pe_partner, delta, prob, my_rand, swap);
#endif              
                }
                // send decision to higher proc
                if (this_global_proc < partner_proc) {
                    MPI_Send(&swap, 1, MPI_INT, partner_proc, 0, MPI_COMM_WORLD);
                } else {
                    MPI_Recv(&swap, 1, MPI_INT, partner_proc, 0, MPI_COMM_WORLD, &status);
                }
            }
            if (swap) {
              // scale velocities
              lammps_scale_velocities(lmp, world2temperature[i_comm], world2temperature[partner_comm]); 
              // reset thermostat temp
              lammps_mod_inst(lmp, 1, fix, "reset_target", &world2temperature[partner_comm]);
              // change replica temperature
              this_replica->temperature = world2temperature[partner_comm]; 
            }

          }
        }
        else {
          printf("Error with dimensions...\n");
          exit(1);
        }

        
        /*-------------------------------------------------------------------------*/
        // Handle any last swapping info stuff here
        if (swap == 1) {
          n_swaps_successful = 1;
          i_replica_id = p_replica_id;
          this_replica->id = i_replica_id;
          // reset temp_id
          i_temp_id = p_temp_id;
          if (dump_swap) {
            // swap dump file names for convenience in post-processing
            if (lammps_get_dump_file(lmp) != NULL) {
              strcpy(my_dumpfile, lammps_get_dump_file(lmp));
              if (this_local_proc == 0) {
                  MPI_Sendrecv(my_dumpfile,      MAXCHARS, MPI_CHAR, partner_proc, 0,
                               partner_dumpfile, MAXCHARS, MPI_CHAR, partner_proc, 0, MPI_COMM_WORLD, &status);
              }
              MPI_Bcast(partner_dumpfile, MAXCHARS, MPI_CHAR, 0, subcomm);
              lammps_change_dump_file(lmp, i_comm, partner_dumpfile);
            } else {
             printf("Error: No dump file specified for MREUS module to swap. Check inputs.\n");
             exit(1);
            }
          }
        }
        // Update lookup tables
        if (this_local_proc == 0){
          MPI_Allgather(&i_replica_id, 1, MPI_INT, world2replicaid, 1, MPI_INT, roots);
          for (i=0; i<i_ncomms; i++) replicaid2world[world2replicaid[i]] = i;
	  MPI_Allgather(&i_temp_id, 1, MPI_INT, world2tempid, 1, MPI_INT, roots);
          MPI_Allgather(&this_replica->temperature, 1, MPI_DOUBLE, world2temperature, 1, MPI_DOUBLE, roots);
        }
        MPI_Bcast(replicaid2world, i_ncomms, MPI_INT, 0, subcomm);
        MPI_Bcast(world2temperature, i_ncomms, MPI_DOUBLE, 0, subcomm);


        /*-------------------------------------------------------------------------*/
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


	// Timer
	MPI_Barrier(MPI_COMM_WORLD);
	double LoopTime_end = MPI_Wtime();

        /*-------------------------------------------------------------------------*/
        // Print status to screen
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
            printf("  Subcomm %d : id = %d\n", s, world2replicaid[s]);
          }
        }


        n_swap_runs_completed++; // update swap runs count

      } // close inner dimensions loop
    } // close outer while loop


/*----------------------------------------------------------------------------------
 *  Some final stats printing 
 */

    if (this_global_proc == 0) {
      printf("\n");
      printf("---------------------------------------------------------------\n");
      printf("Total wall time for MREUS:      %.4f seconds\n", TotalLoopTime);
      printf("Mean wall time b/w swaps:       %.4f seconds\n", TotalLoopTime / (double)(i_nswaps));
      printf("Mean acceptance ratio:          %.3f \n", average_acceptance_ratio / (double)(i_nswaps));
      printf("Total swaps attempted:          %d\n", n_total_swaps_attempted);
      printf("Total swaps successful:         %d\n", n_total_swaps_successful);
      printf("Total acceptance ratio:         %.3f\n", ((double)n_total_swaps_successful) / ((double)n_total_swaps_attempted) );
      printf("---------------------------------------------------------------\n");
    }


/*----------------------------------------------------------------------------------
 * Write final restart file, always 
 */

    if(this_global_proc == 0)
      printf("Writing final restart files.\n");
    MPI_Bcast(world2tempid, i_ncomms, MPI_INT, 0, subcomm);
    char *filename[1];
    char tmp[256]; 
    sprintf(tmp, "restart_final.%s", my_CVID);
    filename[0] = (char*)tmp;
    lammps_write_restart(lmp, filename, timestep);
    MPI_Barrier(MPI_COMM_WORLD);

    // Print the final umbrella window that each replica ended with for restarting
    if (this_local_proc == 0 && i_comm == 0) {
      printf("\nFinal configuration: replica index --> final umbrella window index\n");
      for (p = 0; p < i_ncomms; ++p) {
        printf("Replica %4d --> window %4d\n", p, world2tempid[p]);
      }
      printf("\n\n");
    }
    

/*----------------------------------------------------------------------------------
 * clean it up
 */

    if(this_global_proc == 0) printf("MREUS has completed. Cleaning up memory and exiting...\n");

    lammps_mod_inst(lmp, 3, NULL, "cleanup", NULL);
    lammps_mod_inst(lmp, 0, NULL, "finish", NULL);
    lammps_mod_inst(lmp, 4, NULL, "cleanup", NULL);

    // ** Free the world maps ** // 
    free(world2root);
    free(world2tempid);
    free(replicaid2temp);
    free(world2replicaid);
    free(replicaid2world);
    free(my_dumpfile);
    free(partner_dumpfile);
    free(world2temperature);
}

