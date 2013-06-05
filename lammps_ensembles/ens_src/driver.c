/*
 * driver.c
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "library.h"
#include "replica.h"

int malloc2dchar(char ***array, int n, int m);
int free2dchar(char ***array);


int main(int argc, char **argv) {

/*----------------------------------------------------------------------------------
 * setup MPI
 */
    MPI_Init(&argc,&argv);
    
    int this_global_proc, n_global_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &this_global_proc);
    MPI_Comm_size(MPI_COMM_WORLD, &n_global_procs);


/*----------------------------------------------------------------------------------
 *  Set flag defaults 
 */
    int log_flag = 0;   // Turn on/off print out to log.# files
    int omp_flag = 0;   // Turn on/off OpenMP in LAMMPS/RAPTOR
    int Nstates = 1;    // Number of state partitions
    int Nper = 1;       // Number of procs per partition
    int read_input_flag = 0; // Turn on/off reading LAMMPS input files from file 
    int i;
 
    char input_filename[256]; // Name of input file to read
        
    if (this_global_proc == 0) {

/*----------------------------------------------------------------------------------
 * Print a header to acknowledge the authors 
 */
        printf("\n");
        printf("--------- Multi-replica ensemble driver for LAMMPS v0.1 ---------\n");
        printf(" Brought to you by: Luke Westby, Mladen Rasic, & Adrian W. Lange\n");
        printf("-----------------------------------------------------------------\n\n");

/*----------------------------------------------------------------------------------
 *  Read the input arguments 
 */

        int error = 0;
        int offset = 2;

	// check commandline args for errors
        if (argc <= 1) {
            printf("FLAG ERROR\n");
            printf("---> Not enough command arguments\n");
            error = 1;
        }

        // check if user wants to use OpenMP command line for LAMMPS
        if(strcmp(argv[offset], "-suffix") == 0) { 
          if(strcmp(argv[offset+1], "omp") == 0) { 
            omp_flag = 1;
            printf("---> OpenMP flag detected.\n");
            offset += 2;
          }
        }

        // check if user wants log files printed out
        if(strcmp(argv[offset], "-log") == 0) {
          log_flag = 1; 
          printf("---> Log flag detected. Will print out to numbered log files.\n");
          offset += 1;
        }

        // check if user wants to read LAMMPS input files from disk 
        if(strcmp(argv[offset], "-readinput") == 0) {
          read_input_flag = 1; 
          // Get name of file to read input from
          strcpy( input_filename, argv[offset+1]);
          printf("---> Read input flag detected. Will read input options from file %s\n", input_filename);
          offset += 2;
        }

        int tmp_n_comm = atoi(argv[1]);

        if ((argc != tmp_n_comm + offset) && read_input_flag == 0) { 
            printf("FLAG ERROR\n");
            printf("---> Input files do not match number of instances (%d %d)\n", argc, tmp_n_comm + offset);
            error = 1;
        } 

        if (error == 1) {
          printf("---> Syntax: $ ./bin-name P [-suffix omp] [-log] in.file.1 in.file.2 ... in.file.P\n");
          printf("---> P = number of instances/replicas\n"); 
          printf("---> [-suffix omp] is for optional OpenMP command line argument for LAMMPS\n");
          printf("---> [-log] is for optional printing out to numbered log files for each replica\n\n"); 
          printf("---> Alternatively, you can read LAMMPS inputs from a file and use the following syntax:\n");
          printf("---> Syntax: $ ./bin-name P [-suffix omp] [-log] -readinput [input filename]\n");
          printf("---> The flag -readinput signals that we need to read the inputs from file [input filename]\n");
          printf("\n\nExiting.\n\n");
          exit(1);
        }
        
        printf("Number of processes = %d\n", n_global_procs);
    }

    // Broadcast optional flags and values
    MPI_Bcast(&omp_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&log_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Nstates, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Nper, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&read_input_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

/*----------------------------------------------------------------------------------
 * split MPI_COMM_WORLD into subcomms
 */

    // set up subcomms and lammps pointers
    MPI_Comm subcomm;
    void *lmp;
        
    // grab number of communicators
    int n_comms = atoi(argv[1]);

    MPI_Barrier(MPI_COMM_WORLD);
	    
    if (this_global_proc == 0)
        printf("Number of subcomms = %d\n", n_comms);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // create look-up table for split_key value
    if (this_global_proc == 0) 
        printf("Generating table of instance IDs.\n\n");
    
    int *gproc_lcomm = (int*)malloc(sizeof(int) * n_global_procs);
    int *rk_array    = (int*)malloc(sizeof(int) * 2*n_comms);
    int do_rk = 0; // Do the r-k partitioning?
    char **inputfiles;

    if (read_input_flag) {

      // Handle reading inputs from file 
      malloc2dchar(&inputfiles, n_comms, 256);
      int *npartitions   = (int*)malloc(sizeof(int) * n_comms);
      int *nper_array    = (int*)malloc(sizeof(int) * n_comms);
      ReadInputFiles(gproc_lcomm, inputfiles, n_comms, this_global_proc, npartitions, nper_array, rk_array, input_filename);
      Nstates = npartitions[gproc_lcomm[this_global_proc]];
      Nper = nper_array[gproc_lcomm[this_global_proc]];
      if (rk_array[2*gproc_lcomm[this_global_proc]] > 0) {
        do_rk = 1;
      } 
      // Check correct number of procs
      if (this_global_proc == 0) {
        int sum = 0;
        for (i = 0; i < n_comms; i++) sum += nper_array[i] * npartitions[i];
        if (sum != n_global_procs) {
          printf("Error: The total number of processors in %s is %d.\n", input_filename, sum);
          printf("       The number of processors in MPI_COMM_WORLD is %d\n", n_global_procs);
          printf("       These numbers must match.\n");
          exit(1);
        }
      }
      free(npartitions);
      free(nper_array);

 
    } else {
      // Distribute procs evenly out across partitions
      for (i = 0; i < n_global_procs; i += 1) {
          gproc_lcomm[i] = floor((double)((i * n_comms)/(n_global_procs)));
      }
    }
        
    MPI_Barrier(MPI_COMM_WORLD);
        
    // split
    if (this_global_proc == 0) 
        printf("Splitting MPI_COMM_WORLD\n");
    int split_key = gproc_lcomm[this_global_proc];
    MPI_Comm_split(MPI_COMM_WORLD, split_key, 0, &subcomm);

    // get local rank
    int this_local_proc;
    MPI_Comm_rank(subcomm, &this_local_proc);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (this_local_proc == 0) 
        printf("---> Report from proc 0 on subcomm %d: %d\n", gproc_lcomm[this_global_proc], this_global_proc);
        
/*----------------------------------------------------------------------------------
 * locate input script filename on commandline and disable default logging and screen
 */

    // argv index
    int offset = 2;
    if (omp_flag) offset += 2; 
    if (log_flag) offset += 1;
    int my_file = gproc_lcomm[this_global_proc] + offset;

    // If read inputs from above, modify
    if (read_input_flag) {
      my_file = gproc_lcomm[this_global_proc];
      if (this_local_proc == 0) {
        printf("---> Subcomm %d will use input script: %s\n", gproc_lcomm[this_global_proc], inputfiles[my_file]);
      }
    }
    

/*----------------------------------------------------------------------------------
 * open file and find TEMPER or ANNEAL or COORDX initializer 
 */
            
    MPI_Barrier(MPI_COMM_WORLD);
    if (this_global_proc == 0)
        printf("Attempting to find multi-replica simulation initializer in input script\n\n");


    int nsteps, nevery, sseed, bseed, rate;	// total run, frequency of swap, randswap seed, metropolis seed, cooling rate
    int temperflag = 0, annealflag = 0;		// RE or SA 
    int coordxflag = 0;                         // Coordinate exchange
    double temp, temp_hi, temp_lo;		// RE temp, SA high and low temp
    char fix[50], file[50];			// fix id, restart binary filename
    char CVID[50];			        // Collecvtive Variable ID 
    int replicaID = -1;                         // replica ID for coordx
    int dump_swap = 0;                          // swap dump file names in temper


    // doing everything on root of subcomms
    if (this_local_proc == 0) {

        FILE *infile;
        if (read_input_flag) {
          infile = fopen(inputfiles[my_file], "r");
        } else {
          infile = fopen(argv[my_file], "r");
        }

		// buffer and file position for scanning
		char command[9];
		fpos_t position;

		// search for TEMPER or ANNEAL line by checking first 9 chars
		while(temperflag == 0 && annealflag == 0 && coordxflag == 0 
                      && !feof(infile)) {	
			fgetpos(infile, &position);	// store position
			fgets(command, 9, infile);	// read in 9 chars
			fscanf(infile, "\n");		// move to end of line
			// check
			if     (strcmp(command, "#TEMPER:") == 0) temperflag = 1;
			else if(strcmp(command, "#ANNEAL:") == 0) annealflag = 1;
			else if(strcmp(command, "#COORDX:") == 0) coordxflag = 1;
		}

		// come back to beginning of line
		fsetpos(infile, &position);

		// scan values from line and clip commas off char *'s
		if(temperflag) {
			int n = fscanf(infile, "#TEMPER: run %d, swap %d, temp %lf, fix %s seed %d, dumpswap %d", 
                                       &nsteps, &nevery, &temp, fix, &sseed, &dump_swap);
                        if (n != 6) {
                          printf("%d %d %lf %s %d %d\n", nsteps, nevery, temp, fix, sseed, dump_swap);
                          printf("Problem reading #TEMPER line. Please check that formatting strictly complies.\n");
                          exit(1);
                        }
			int len = strlen(fix) - 1;
			fix[len] = 0;
		} else if(annealflag) {
			fscanf(infile, "#ANNEAL: run %d, swap %d, rate %d, file %s fix %s seed %d, temp_hi %lf, temp_lo %lf\n", 
                               &nsteps, &nevery, &rate, file, fix, &bseed, &temp_hi, &temp_lo);
			int len_fix = strlen(fix) - 1;
			int len_file = strlen(file) -1;
			fix[len_fix] = 0;
			file[len_file] = 0;
		} else if(coordxflag) {
		      fscanf(infile, "#COORDX: fix %s seed %d", fix, &sseed);
		      int len_fix = strlen(fix) - 1;
		      fix[len_fix] = 0;
                      // search for replica line for replica id
		      while( coordxflag == 1 && !feof(infile) ) {	
                        int tmp1, tmp3;
                        double tmp2;
			fgetpos(infile, &position);	// store position
			fgets(command, 10, infile);	// read in 10 chars
			fscanf(infile, "\n");		// move to end of line
			if ( strcmp(command, "#REPLICA:") == 0) { 
		          // come back to beginning of line
		          fsetpos(infile, &position);
  			  if ( fscanf(infile, "#REPLICA: id %d, ndim %d, temp %lf, tdim %d", 
                                      &replicaID, &tmp1, &tmp2, &tmp3) == 4) {
                            coordxflag = 2;
                          }
                        }
                      }
                }
		
        fclose(infile);
    }

    // bcast flags and common values
    MPI_Bcast(&temperflag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&annealflag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&coordxflag, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
    MPI_Bcast(&nsteps, 1, MPI_INT, 0, subcomm);
    MPI_Bcast(&nevery, 1, MPI_INT, 0, subcomm);
    MPI_Bcast(&bseed, 1, MPI_INT, 0, subcomm);
    MPI_Bcast(fix, 50, MPI_CHAR, 0, subcomm);
    MPI_Bcast(&dump_swap, 1, MPI_INT, 0, subcomm); 

	// print for error checking and bcast specific values
	if(temperflag) {
		
	    MPI_Bcast(&sseed, 1, MPI_INT, 0, subcomm); 
	    if(this_global_proc == 0) {
		printf("Preparing to run replica exchange:\n");
	        printf("---> Run %d total timesteps\n", nsteps);
	        printf("---> Attempt exchange every %d timesteps\n", nevery);
	        printf("---> Using fix id %s\n", fix);
	        printf("---> Using random swap seed %d\n", sseed);
	    } if(this_local_proc == 0)
		printf("---> Instance %d using temp %lf\n", split_key, temp);
	    MPI_Bcast(&temp, 1, MPI_DOUBLE, 0, subcomm);

	} else if(annealflag) {
	
	    if(this_global_proc == 0) {
		printf("Preparing to run simulated annealing:\n");
	        printf("---> Run %d total timesteps\n", nsteps);
	        printf("---> Attempt exchange every %d timesteps\n", nevery);
		printf("---> Using metroplis seed %d + (global rank)\n", bseed);
	        printf("---> Decrease temperature every %d timesteps\n", rate);
	        printf("---> Using fix id %s\n", fix);
		printf("---> Starting temp: %lf\n", temp_hi);
		printf("---> Ending temp: %lf\n", temp_lo);
		} if(this_local_proc == 0)
			printf("---> Instance %d using restart binary filename %s\n", split_key, file);

		MPI_Bcast(file, strlen(file), MPI_INT, 0, subcomm);
		MPI_Bcast(&rate, 1, MPI_INT, 0, subcomm);
		MPI_Bcast(&temp_hi, 1, MPI_DOUBLE, 0, subcomm);
		MPI_Bcast(&temp_lo, 1, MPI_DOUBLE, 0, subcomm);

	} else if(coordxflag) {
		
	    MPI_Bcast(&sseed, 1, MPI_INT, 0, subcomm); 
	    if(this_global_proc == 0) {
		printf("Preparing to run replica exchange with coordinate exchange:\n");
	        printf("---> Using fix id %s\n", fix);
	        printf("---> Using random swap seed %d\n", sseed);
	    } 
            MPI_Bcast(&replicaID, 1, MPI_INT, 0, subcomm);

	} else {		// could not find TEMPER or ANNEAL or COORDX 

		if(this_global_proc == 0) {
			printf("No multi-replica simulation specificied in input script.\n");
			printf("Please specifiy a simulation.\n");
                        printf("Valid options are (whitespace sensitive):\n");
                        printf("'#TEMPER: ', '#ANNEAL: ', '#COORDX: '\n"); 
			printf("Exiting.\n\n");
			exit(1);
		}
	}

/*----------------------------------------------------------------------------------
 *  Handle passing command line input options 
 */

    MPI_Barrier(MPI_COMM_WORLD);
    if (this_global_proc == 0) {
     	printf("\n");
        printf("Disabling default LAMMPS logging and screen output\n\n");
    }

    // AWGL : have to put in command line argument options here

    // hardcode lammps args	
    char str1[] = "./ens_driver";
    char strin[] = "-in";
    char strfile[256];
    if (read_input_flag) {
      sprintf(strfile,"%s",inputfiles[my_file]); 
    } else {
      sprintf(strfile,"%s",argv[my_file]); 
    }
    char str2[] = "-log";
    char str3_none[] = "none";
    char str3_log[256];
    //if (reusflag) {
    //  // New CVID labelling
    //  sprintf(str3_log,"log.cv.%s", CVID);
    //} else {
    if (coordxflag) {
      sprintf(str3_log,"log.id.%d", replicaID);
    } else {
      // Default to split key labelling
      sprintf(str3_log,"log.%d", split_key);
    }
    char str4[] = "-screen";
    char str_suffix[]  = "-suffix";
    char str_omp[] = "omp";
    char str_partition[] = "-partition";
    char str_nstates_nper[256];
    sprintf(str_nstates_nper,"%dx%d", Nstates, Nper);

    // ** Build list ** //
    char *arglist[13];
    int n = 0;
    // Standard commands
    arglist[n++] = str1;
    arglist[n++] = strin;
    arglist[n++] = strfile;
    arglist[n++] = str2;
    if (log_flag) arglist[n++] = str3_log;
    else          arglist[n++] = str3_none;
    arglist[n++] = str4;
    arglist[n++] = str3_none;
    // Optional commands
    if (omp_flag) {
      arglist[n++] = str_suffix;
      arglist[n++] = str_omp;
    }
    if (read_input_flag && Nstates > 1 && do_rk == 0) {
      // Only do the partitioning if more than one state requested
      arglist[n++] = str_partition;
      arglist[n++] = str_nstates_nper;
    }
    if (read_input_flag && do_rk) {
      arglist[n++] = str_partition;
      char str_tmp[256];
      sprintf(str_tmp,"%d", Nstates); 
      arglist[n++] = str_tmp;
      sprintf(str_tmp,"%d", rk_array[2*gproc_lcomm[this_global_proc]]); 
      arglist[n++] = str_tmp;
      sprintf(str_tmp,"%d", rk_array[2*gproc_lcomm[this_global_proc]+1]); 
      arglist[n++] = str_tmp;
    }

    // create pointer
    char** args = arglist;

/*----------------------------------------------------------------------------------
 * open LAMMPS - assign LAMMPS pointer, read input script, run simulation
 */

    MPI_Barrier(MPI_COMM_WORLD);

    int num_args = 7;
    if (omp_flag) num_args += 2; 
    if (read_input_flag && Nstates > 1 && do_rk == 0) num_args += 2; 
    if (read_input_flag && do_rk) num_args += 3;

    if(this_local_proc == 0) {
      printf("---> Opening LAMMPS on subcomm %d with command:", split_key);
      int j;
      for (j=0; j<num_args; j++)
        printf("%s ", args[j]);
      printf("\n");
    }

    // assign lmp to new LAMMPS instance	
    lammps_open(num_args, args, subcomm, &lmp);           

    MPI_Barrier(MPI_COMM_WORLD);
    if (this_local_proc == 0) {
      if (read_input_flag) {
        printf("---> Opening LAMMPS script %s subcomm %d...\n", inputfiles[my_file], split_key);
      } else {
        printf("---> Opening LAMMPS script %s subcomm %d...\n", argv[my_file], split_key);        
      }
    }

    // open input script
    if (read_input_flag) {
      lammps_file(lmp, inputfiles[my_file]);
    } else {
      lammps_file(lmp, argv[my_file]);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int skip_run = 0;
    // run simulation
    if(temperflag) {
    	if (this_global_proc == 0) 
    	    printf("---> Beginning replica exchange...\n\n"); 

        if (!skip_run)
          temper(lmp, subcomm, nsteps, nevery, n_comms, split_key, temp, fix, sseed, dump_swap); 
    }
    else if(annealflag) {
       	if(this_global_proc == 0)
    	  printf("---> Beginning simulated annealing...\n\n");
//		anneal(lmp, subcomm, file, fix, bseed, split_key, n_comms, nsteps, nevery, rate, temp_hi, temp_lo);
    }
    else if(coordxflag) {
 
        // ** Set up the replica data structure here ** //
        Replica this_replica;
        if (read_input_flag) {
          ReadReplica(&this_replica, subcomm, split_key, this_local_proc, inputfiles[my_file]);
        } else {
          ReadReplica(&this_replica, subcomm, split_key, this_local_proc, argv[my_file]);
        }

    	if (this_global_proc == 0) 
    	    printf("---> Beginning replica exchange with coordinates...\n\n"); 

        if (!skip_run)
          coord_exchange(lmp, subcomm, n_comms, split_key, &this_replica, fix, sseed); 

        // ** Free replica array ** //
        free(this_replica.neighbors);
        free(this_replica.dim_run);
        free(this_replica.dim_nevery);
    }

/*----------------------------------------------------------------------------------
 * close LAMMPS and clean up
 */

    MPI_Barrier(MPI_COMM_WORLD);
    if (this_local_proc == 0) 
       printf("---> Closing LAMMPS on subcomm %d.\n", gproc_lcomm[this_global_proc]);

    free(gproc_lcomm);
    if (read_input_flag) {
      free2dchar(&inputfiles);
    }
    free(rk_array);

    lammps_close(lmp);

    MPI_Barrier(MPI_COMM_WORLD);

/*----------------------------------------------------------------------------------
 * Print a footer 
 */
    if (this_global_proc == 0) {
        printf("\n");
        printf("-----------------------------------------------------------------\n");
        printf("                         Finished!                               \n");
        printf("-----------------------------------------------------------------\n\n");
    }

    MPI_Finalize();

    return 0;

}


/*----------------------------------------------------------------------------------
 * 2d array helper functions 
 */

int malloc2dchar(char ***array, int n, int m) {
    char *p = (char *)malloc(n*m*sizeof(char));
    (*array) = (char **)malloc(n*sizeof(char*));
    int i;
    for (i=0; i<n; i+=1)
       (*array)[i] = &(p[i*m]);
    return 0;
}

int free2dchar(char ***array) {
    free(&((*array)[0][0]));
    free(*array);
    return 0;
}
