/*
 * anneal.c
 * This file is part of lammps-ensembles
 *
 * Copyright (C) 2012 - Luke Westby
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

#include "mpi.h"
#include "library.h"
#include "replica.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


void anneal(void *lmp, MPI_Comm subcomm, char *restart_name, char *fix, int seed, int comm, int ncomms, int nsteps, int nevery, int rate, double t_max, double t_min) {

/*----------------------------------------------------------------------------------
 * initialize variables
 */

	// grab values
 	int i_nsteps = nsteps;	// number of total steps to run for
 	int i_nevery = nevery;	// number of steps per annealing run
 	int i_rate = rate;		// number of steps per temperature adjustment (must be from 1 to nevery/2)
 	int i_max_t = t_max;		// starting temperature
 	int i_min_t = t_min;		// ending temperature
 	int i_seed = seed;		// rand seed
	int i_comm = comm;		// which instance am i
	int i_ncomms = ncomms;	// how many instances

	// check value of i_rate
	if(i_rate < 1 || i_rate > i_nevery / 2)
		exit(1);

	// check compatability of frequency to length
	if(i_nsteps % i_nevery != 0) 
		exit(1);

/*----------------------------------------------------------------------------------
 * MPI stuff
 */

	// local
	int this_proc, n_procs;
	MPI_Comm_rank(subcomm, &this_proc);
	MPI_Comm_size(subcomm, &n_procs);

	// global
	int this_global_proc;
	MPI_Comm_rank(MPI_COMM_WORLD, &this_global_proc);

/*----------------------------------------------------------------------------------
 * create comm for roots on subcomms
 */

	int split_key;
	MPI_Comm roots;
	if(this_proc == 0) split_key = 0;
	else split_key = 1;
	MPI_Comm_split(MPI_COMM_WORLD, split_key, 0, &roots);
		
/*----------------------------------------------------------------------------------
 * set up array of restart filenames on master
 */

	// initialize restart id as instance id
	int i_restart = i_comm;
	
	// build array
	char id2restart[i_ncomms][MAX_FILE_LEN];
	if(this_proc == 0)
		MPI_Allgather(&restart_name, MAX_FILE_LEN, MPI_CHAR, id2restart, MAX_FILE_LEN, MPI_CHAR, roots);
	MPI_Bcast(id2restart, MAX_FILE_LEN * i_ncomms, MPI_CHAR, 0, subcomm);

	// write to file
	char *i_write = command_generate(id2restart[i_comm], 1);
	lammps_command(lmp, i_write);
	free(i_write);

/*----------------------------------------------------------------------------------
 * set up array returning global proc of root of a given instance
 */

	int instance2root[i_ncomms];
	if(this_proc == 0)
		MPI_Allgather(&this_global_proc, 1, MPI_INT, instance2root, 1, MPI_INT, roots);
	MPI_Bcast(instance2root, i_ncomms, MPI_INT, 0, subcomm);

/*----------------------------------------------------------------------------------
 * warm up random number generator
 */

	int warmup;
	for (warmup = 0; warmup < 100; warmup++)
		rng(&i_seed);

/*----------------------------------------------------------------------------------
 * get boltzmann constant
 */

	double *boltz_ptr = (double *)lammps_extract_global(lmp, "boltz");
	double boltz = *boltz_ptr;

/*----------------------------------------------------------------------------------
 * Setup update, init lammps, setup integrator
 */

	lammps_mod_inst(lmp, 4, NULL, "setup", NULL);	// update things (see library.cpp)
	lammps_mod_inst(lmp, 0, NULL, "init", NULL);		// lmp->init()
	lammps_mod_inst(lmp, 3, NULL, "setup", NULL);	// lmp->update->integrate->setup()

/*----------------------------------------------------------------------------------
 * Lets simulate some annealing and whatnot
 */

	/* variables ---------------------------------------------------------------- */
		
	int n_iter, this_iter, n_cool_iter, this_cool_iter;
	int swapflag;
	double criterion, old_temp, i_temp = i_max_t, i_range_t = i_max_t - i_min_t;
	double *old_pe, *new_pe;
	char *i_read;
	
	// global iterations
	n_iter = i_nsteps / i_nevery;
	this_iter = 0;

	// cooling iterations
	n_cool_iter = i_nevery / i_rate;
	this_cool_iter = 0;

	/* setup initial state ------------------------------------------------------ */
	
	// packaging pe and instance id for MPI_Reduce
	struct {
		double value;
		int id;
	} pe_in, pe_out;
	pe_out.id = i_comm;

	/* run ---------------------------------------------------------------------- */
 	for(this_iter = 0; this_iter < n_iter; this_iter++) {

		// cooling step
		for(this_cool_iter = 0; this_cool_iter < n_cool_iter; this_cool_iter++) {

			// calculate pre-run pe
			old_pe = (double *)lammps_extract_compute(lmp, "thermo_pe", 0, 0);

			// store temp from last iteration and get temp for this iteration
			old_temp = i_temp;
			i_temp = i_max_t - this_cool_iter * (i_range_t)/n_cool_iter;

			// if temperature cooled, slow molecules down and reset thermostat
			if(i_temp != old_temp) {
				lammps_scale_velocities(lmp, old_temp, i_temp);
				lammps_mod_inst(lmp, 1, fix, "reset_target", &i_temp);
			}
		
			// add compute step
			lammps_mod_inst(lmp, 2, "thermo_pe", "addstep", &i_nevery);

			// run for n_steps
			lammps_mod_inst(lmp, 3, NULL, "run", &i_nsteps);

			// get new pe
			new_pe = (double *)lammps_extract_compute(lmp, "thermo_pe", 0, 0);

			// root proc does some decision making on the run
			if(this_proc == 0) {
				//calculate metropolis
				criterion = (*old_pe - *new_pe) / (boltz * i_temp);
				if(criterion >= 0.0) swapflag = 1;
				else if(rng(&i_seed) < exp(criterion)) swapflag = 1;
				else swapflag = 0;
			}	

			// broadcast decision
			MPI_Bcast(&swapflag, 1, MPI_INT, 0, subcomm);

			// check if swap occured
			// if no, read last restart file and start over
			// if yes, save current state to restart file
			if(swapflag == 0) {
				i_read = command_generate(id2restart[i_comm], 0);
				lammps_command(lmp, i_read);
				free(i_read);
				pe_out.value = *old_pe;
			} else {
				i_write = command_generate(id2restart[i_comm], 0);
				lammps_command(lmp, i_write);
				free(i_write);
				pe_out.value = *new_pe;
			}

		}

		// check and exchange step
		
		// master proc finds lowest pe
		if(this_global_proc == 0) {
			MPI_Reduce(&pe_out, &pe_in, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, roots);
			i_restart = pe_in.id;
		}

		// Bcast new restart point
		MPI_Bcast(&i_restart, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// Read restart 
		i_read = command_generate(id2restart[i_restart], 0);
		lammps_command(lmp, i_read);
		free(i_read);

	}

/*----------------------------------------------------------------------------------
 * clean it up and finish
 */

	lammps_mod_inst(lmp, 3, NULL, "cleanup", NULL);
    lammps_mod_inst(lmp, 0, NULL, "finish", NULL);
    lammps_mod_inst(lmp, 4, NULL, "cleanup", NULL);

}

char *command_generate(char *filename, int type) {

	size_t len1 = strlen(filename);

	if(type == 0) {
		char *read_name = "read_restart";
		size_t lenr = strlen(read_name);
		char *command_read = (char *)malloc((len1 + lenr + 2) * sizeof(char));
		strcpy(command_read, read_name);
		//command_read[len1] = "\t";
		strcpy(&command_read[len1 + 1], filename);
		return command_read;

	} else if(type == 1) {
		char *write_name = "write_restart";
		size_t lenw = strlen(write_name);
		char *command_write = (char *)malloc((len1 + lenw + 2) * sizeof(char));
		strcpy(command_write, write_name);
		//command_write[len1] = "\t";
		strcpy(&command_write[len1 + 1], filename);
		return command_write;

	} else return NULL;

}
