/*
 * rng2.c
 * This file is part of lammps-ensemble
 *
 * Copyright (C) 2012 - Mladen Rasic & Luke Westby
 *
 * lammps-ensemble is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * lammps-ensemble is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with lammps-ensemble; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, 
 * Boston, MA  02110-1301  USA
 */

// Reads file named "input.dat" for a list of LAMMPS input scripts to use for each subcommunicator 

#include "replica.h"
#include "stdlib.h"
#include "string.h"

#define MAX_LENGTH 1024

void ReadInputFiles(int *gproc_lcomm, char **inputfiles, int n_comms, int this_global_proc,
                    int *partitions, int *nper, int *rk, char *input_filename)
{
      // File syntax:
      // (Comm index) (Filename) (# procs in comm) (# partitions) [# real procs] [# k procs]
      // () = Mandatory, [] = optional
      
      int i;
      FILE *fp;

      if (this_global_proc == 0) {
        printf("---> Attempting to read input files from file %s ...\n", input_filename);
        fp = fopen(input_filename, "r");
        if (fp == NULL) {
          printf("Failure to read from file %s\n", input_filename);
          fprintf(stderr, "Failure to read from file %s\n", input_filename);
          exit(1);
        }
      }

      char fname[256];
      char line[MAX_LENGTH];
      int commind, n_procs_in_comm, n_partitions, n_r, n_k;
      int send_rk = 0;
      int gprocnum = 0;

      for (i=0; i<n_comms; i+=1) {

        if (this_global_proc == 0) {
          if ( fgets(line, MAX_LENGTH, fp) != NULL) {
            // Check for usual 
            if ( sscanf(line, "%d %s %d %d", &commind, fname, &n_procs_in_comm, &n_partitions) == 4 ) {
              send_rk = 0;
            }
            // Usual, but ignore partitioning for ease
            else if ( sscanf(line, "%d %s %d", &commind, fname, &n_procs_in_comm) == 3 ) {
              n_partitions = 1;
              send_rk = 0;
            }
            // Check for r-k partitioning
            else if ( sscanf(line, "%d %s %d %d %d %d", 
                      &commind, fname, &n_procs_in_comm, &n_partitions, &n_r, &n_k) == 6 ) {
              if (n_partitions != 2) {
                printf("Must have 2 paritions for r-k paritioning in verlet/split. Comm %d: partitions %d\n", 
                       commind, n_partitions);
                exit(1);
              }
              send_rk = 1;
            }
            // Error checking of input
            if (commind > n_comms || commind < 0) {
              printf("Attempted to use %d index for %d subcommunicators. Crash.\n", commind, n_comms);
              fprintf(stderr, "Attempted to use %d index for %d subcommunicators. Crash.\n", commind, n_comms);
              exit(1);
            }
            if (n_partitions < 1) {
              printf("Cannot have less than one partition. Comm %d: partitions %d\n", commind, n_partitions);
              exit(1);
            }
            if (n_procs_in_comm < 1) {
              printf("Cannot have less than one proc in comm. Comm %d: procs %d\n", commind, n_procs_in_comm);
              exit(1);
            }
          }
        }

        // Broadcast information from line
        MPI_Bcast(&commind, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(fname, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(&n_procs_in_comm, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&n_partitions, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&send_rk, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        strcpy( inputfiles[commind], fname );
        partitions[commind] = n_partitions;
        nper[commind] = n_procs_in_comm / n_partitions;
        int j;
        for (j=0; j < n_procs_in_comm; j+=1) {
          gproc_lcomm[gprocnum] = commind;
          gprocnum += 1;
        }

        // Handle possible r-k partition
        if (send_rk) {
          MPI_Bcast(&n_r, 1, MPI_INT, 0, MPI_COMM_WORLD);
          MPI_Bcast(&n_k, 1, MPI_INT, 0, MPI_COMM_WORLD);
          rk[2*commind]   = n_r;
          rk[2*commind+1] = n_k;
        }
        else {
          rk[2*commind]   = 0;
          rk[2*commind+1] = 0;
        }

      }

      if (this_global_proc == 0) {
        fclose(fp);
        printf("---> Finished reading %s.\n", input_filename);
      }
}
