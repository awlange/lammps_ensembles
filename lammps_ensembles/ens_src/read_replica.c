/*
 * read_replica.c 
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

void ReadReplica(Replica *this_replica, MPI_Comm subcomm, int split_key, int this_local_proc, char *input_filename)
{
      // File syntax:
      // () = Mandatory, [] = optional
      // #REPLICA: (id) (N_dimensions) (temperature) (temp_dim) 
      // #NEIGHBORS: (index of dimension) (minus neighbor) (plus neighbor) 
      
      // Only local rank master reads file 
      FILE *fp;
      if (this_local_proc == 0) {
        printf("---> Subcomm %d reading replica information from file %s ...\n", split_key, input_filename);
        fp = fopen(input_filename, "r");
        if (fp == NULL) {
          printf("Failure to read from file %s\n", input_filename);
          fprintf(stderr, "Failure to read from file %s\n", input_filename);
          exit(1);
        }
      }
 
      Replica tmp;
      tmp.N_dimensions = 0;
      tmp.reus_dim = -1;
      tmp.temp_dim = -1;
      tmp.lambda_dim = -1;
      int i;
      int dim = -1;
      int min_neigh, plus_neigh;
      int num;

      char line[MAX_LENGTH];
      char subline[MAX_LENGTH];
      if (this_local_proc == 0) {
        while ( !feof(fp) ) {
          if ( fgets(line, MAX_LENGTH, fp) != NULL) {
            int run, swap;
            if ( sscanf(line, "#REPLICA: id %d, ndim %d, temp %lf, tdim %d", 
                        &tmp.id, 
                        &tmp.N_dimensions, 
                        &tmp.temperature, 
                        &tmp.temp_dim) == 4 ) {
              // Allocate memory 
              tmp.neighbors  = (int*)malloc( sizeof(int) * tmp.N_dimensions * 2 ); // 2 neighbors in each dimension
              tmp.dim_run    = (int*)malloc( sizeof(int) * tmp.N_dimensions ); 
              tmp.dim_nevery = (int*)malloc( sizeof(int) * tmp.N_dimensions ); 
              tmp.dim_num    = (int*)malloc( sizeof(int) * tmp.N_dimensions ); 
            }
            else if ( sscanf(line, "#DIMENSION: %d num %d run %d swaps %d", &dim, &num, &run, &swap) == 4 ) {
              // DIMENSION line using number of swaps
              tmp.dim_run[dim] = run;
              tmp.dim_nevery[dim] = run / swap;
              tmp.dim_num[dim] = num;
            }
            else if ( sscanf(line, "#DIMENSION: %d num %d run %d swapfreq %d", &dim, &num, &run, &swap) == 4 ) {
              // DIMENSION line using swap frequency
              tmp.dim_run[dim] = run;
              tmp.dim_nevery[dim] = swap;
              tmp.dim_num[dim] = num;
              // check that swap divides run
              if (run % swap) {
                printf("Error with swap frequency. Does not divide run. %d %% %d = %d\n", run, swap, run % swap);
                fprintf(stderr, "Error with swap frequency. Does not divide run. %d %% %d = %d\n", run, swap, run % swap);
                exit(1);
              }
            }
            else if ( sscanf(line, "#NEIGHBORS: %d %d %d", &dim, &min_neigh, &plus_neigh) == 3 ) {
              tmp.neighbors[2*dim  ] = min_neigh;  
              tmp.neighbors[2*dim+1] = plus_neigh;  
            }
          }
        }
        if (tmp.N_dimensions < 1) {
          printf("Error with number of dimensions in file %s\n", input_filename);
          fprintf(stderr, "Error with number of dimensions in file %s\n", input_filename);
          exit(1);
        }
        fclose(fp);
        printf("Reading replica information complete on subcomm %d\n", split_key);
      }
      MPI_Barrier(subcomm);


      // Broadcast replica data 
      MPI_Bcast(&tmp.id,           1, MPI_INT,    0, subcomm);
      MPI_Bcast(&tmp.N_dimensions, 1, MPI_INT,    0, subcomm);
      MPI_Bcast(&tmp.temperature,  1, MPI_DOUBLE, 0, subcomm);
      MPI_Bcast(&tmp.temp_dim,     1, MPI_INT,    0, subcomm);

      // Copy over from tmp 
      this_replica->id           = tmp.id;
      this_replica->N_dimensions = tmp.N_dimensions;
      this_replica->temperature  = tmp.temperature;
      this_replica->temp_dim     = tmp.temp_dim;
      this_replica->neighbors    = (int*)malloc( sizeof(int) * tmp.N_dimensions * 2 ); // 2 neighbors in each dimension
      this_replica->dim_run      = (int*)malloc( sizeof(int) * tmp.N_dimensions ); 
      this_replica->dim_nevery   = (int*)malloc( sizeof(int) * tmp.N_dimensions ); 
      this_replica->dim_num      = (int*)malloc( sizeof(int) * tmp.N_dimensions ); 
      if (this_local_proc == 0) {
        for (i=0; i<tmp.N_dimensions; ++i) {
          this_replica->neighbors[2*i  ] = tmp.neighbors[2*i  ];
          this_replica->neighbors[2*i+1] = tmp.neighbors[2*i+1];
          this_replica->dim_run[i]       = tmp.dim_run[i];
          this_replica->dim_nevery[i]    = tmp.dim_nevery[i];
          this_replica->dim_num[i]       = tmp.dim_num[i];
        }
      }
      // Broadcast the array data
      MPI_Bcast(this_replica->neighbors, 2*tmp.N_dimensions, MPI_INT, 0, subcomm);
      MPI_Bcast(this_replica->dim_run,     tmp.N_dimensions, MPI_INT, 0, subcomm);
      MPI_Bcast(this_replica->dim_nevery,  tmp.N_dimensions, MPI_INT, 0, subcomm);
      MPI_Bcast(this_replica->dim_num,     tmp.N_dimensions, MPI_INT, 0, subcomm);

      // Free memory for tmp
      if (this_local_proc == 0) {
        free(tmp.neighbors);
        free(tmp.dim_run);
        free(tmp.dim_nevery);
        free(tmp.dim_num);
      }
}


/* ----------------------------------------------------------------------- */
// Slightly different routine for MREUS

void ReadReplicaMREUS(Replica *this_replica, MPI_Comm subcomm, int split_key, int this_local_proc, char *input_filename)
{
      // File syntax:
      // () = Mandatory, [] = optional
      // #REPLICA: (id) (N_dimensions) (temperature) 
      // #NEIGHBORS: (index of dimension) (minus neighbor) (plus neighbor) 
      
      // Only local rank master reads file 
      FILE *fp;
      if (this_local_proc == 0) {
        printf("---> Subcomm %d reading replica information from file %s ...\n", split_key, input_filename);
        fp = fopen(input_filename, "r");
        if (fp == NULL) {
          printf("Failure to read from file %s\n", input_filename);
          fprintf(stderr, "Failure to read from file %s\n", input_filename);
          exit(1);
        }
      }
 
      Replica tmp;
      tmp.N_dimensions = 0;
      tmp.reus_dim = -1;
      tmp.temp_dim = -1;
      tmp.lambda_dim = -1;
      int i;
      int dim = -1;
      int min_neigh, plus_neigh;
      int num;

      char line[MAX_LENGTH];
      char subline[MAX_LENGTH];
      if (this_local_proc == 0) {
        while ( !feof(fp) ) {
          if ( fgets(line, MAX_LENGTH, fp) != NULL) {
            int run, swap, type;
            if ( sscanf(line, "#REPLICA: id %d, ndim %d, temp %lf", 
                        &tmp.id, 
                        &tmp.N_dimensions, 
                        &tmp.temperature) == 3 ) {
              // Allocate memory 
              tmp.neighbors  = (int*)malloc( sizeof(int) * tmp.N_dimensions * 2 ); // 2 neighbors in each dimension
              tmp.dim_run    = (int*)malloc( sizeof(int) * tmp.N_dimensions ); 
              tmp.dim_nevery = (int*)malloc( sizeof(int) * tmp.N_dimensions ); 
              tmp.dim_num    = (int*)malloc( sizeof(int) * tmp.N_dimensions ); 
            }
            else if ( sscanf(line, "#DIMENSION: %d num %d type %d run %d swaps %d", &dim, &num, &type, &run, &swap) == 5 ) {
              // DIMENSION line using number of swaps
              tmp.dim_run[dim] = run;
              tmp.dim_nevery[dim] = run / swap;
              tmp.dim_num[dim] = num;
              printf("type = %d dim = %d\n", type, dim);
              if      (type == DIM_REUS)   tmp.reus_dim = dim;
              else if (type == DIM_TEMPER) tmp.temp_dim = dim;
              else if (type == DIM_LAMBDA) tmp.lambda_dim = dim;
            }
            else if ( sscanf(line, "#DIMENSION: %d num %d type %d run %d swapfreq %d", &dim, &num, &type, &run, &swap) == 5 ) {
              // DIMENSION line using swap frequency
              tmp.dim_run[dim] = run;
              tmp.dim_nevery[dim] = swap;
              tmp.dim_num[dim] = num;
              // check that swap divides run
              if (run % swap) {
                printf("Error with swap frequency. Does not divide run. %d %% %d = %d\n", run, swap, run % swap);
                fprintf(stderr, "Error with swap frequency. Does not divide run. %d %% %d = %d\n", run, swap, run % swap);
                exit(1);
              }
              if      (type == DIM_REUS)   tmp.reus_dim = dim;
              else if (type == DIM_TEMPER) tmp.temp_dim = dim;
              else if (type == DIM_LAMBDA) tmp.lambda_dim = dim;
            }
            else if ( sscanf(line, "#NEIGHBORS: %d %d %d", &dim, &min_neigh, &plus_neigh) == 3 ) {
              tmp.neighbors[2*dim  ] = min_neigh;  
              tmp.neighbors[2*dim+1] = plus_neigh;  
            }
          }
        }
        if (tmp.N_dimensions < 1) {
          printf("Error with number of dimensions in file %s\n", input_filename);
          fprintf(stderr, "Error with number of dimensions in file %s\n", input_filename);
          exit(1);
        }
        fclose(fp);
        printf("Reading replica information complete on subcomm %d\n", split_key);
      }
      MPI_Barrier(subcomm);


      // Broadcast replica data 
      MPI_Bcast(&tmp.id,           1, MPI_INT,    0, subcomm);
      MPI_Bcast(&tmp.N_dimensions, 1, MPI_INT,    0, subcomm);
      MPI_Bcast(&tmp.temperature,  1, MPI_DOUBLE, 0, subcomm);
      MPI_Bcast(&tmp.temp_dim,     1, MPI_INT,    0, subcomm);
      MPI_Bcast(&tmp.reus_dim,     1, MPI_INT,    0, subcomm);
      MPI_Bcast(&tmp.lambda_dim,   1, MPI_INT,    0, subcomm);

      // Copy over from tmp 
      this_replica->id           = tmp.id;
      this_replica->N_dimensions = tmp.N_dimensions;
      this_replica->temperature  = tmp.temperature;
      this_replica->temp_dim     = tmp.temp_dim;
      this_replica->reus_dim     = tmp.reus_dim;
      this_replica->lambda_dim   = tmp.lambda_dim;
      this_replica->neighbors    = (int*)malloc( sizeof(int) * tmp.N_dimensions * 2 ); // 2 neighbors in each dimension
      this_replica->dim_run      = (int*)malloc( sizeof(int) * tmp.N_dimensions ); 
      this_replica->dim_nevery   = (int*)malloc( sizeof(int) * tmp.N_dimensions ); 
      this_replica->dim_num      = (int*)malloc( sizeof(int) * tmp.N_dimensions ); 
      if (this_local_proc == 0) {
        for (i=0; i<tmp.N_dimensions; ++i) {
          this_replica->neighbors[2*i  ] = tmp.neighbors[2*i  ];
          this_replica->neighbors[2*i+1] = tmp.neighbors[2*i+1];
          this_replica->dim_run[i]       = tmp.dim_run[i];
          this_replica->dim_nevery[i]    = tmp.dim_nevery[i];
          this_replica->dim_num[i]       = tmp.dim_num[i];
        }
      }
      // Broadcast the array data
      MPI_Bcast(this_replica->neighbors, 2*tmp.N_dimensions, MPI_INT, 0, subcomm);
      MPI_Bcast(this_replica->dim_run,     tmp.N_dimensions, MPI_INT, 0, subcomm);
      MPI_Bcast(this_replica->dim_nevery,  tmp.N_dimensions, MPI_INT, 0, subcomm);
      MPI_Bcast(this_replica->dim_num,     tmp.N_dimensions, MPI_INT, 0, subcomm);

#ifdef MREUS_DEBUG
      // Print the dimension types
      if (this_local_proc == 0) {
        printf("Comm %d Dimension types:\n", split_key);
        printf("Comm %d Dimension %d is the REUS dimension\n", split_key, this_replica->reus_dim);
        printf("Comm %d Dimension %d is the TEMPER dimension\n", split_key, this_replica->temp_dim);
        printf("Comm %d Dimension %d is the RELAMBDA dimension\n", split_key, this_replica->lambda_dim);
      }
#endif

      // Free memory for tmp
      if (this_local_proc == 0) {
        free(tmp.neighbors);
        free(tmp.dim_run);
        free(tmp.dim_nevery);
        free(tmp.dim_num);
      }
}
