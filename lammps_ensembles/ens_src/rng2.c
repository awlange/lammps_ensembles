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

// AWGL: I don't trust that the other random number generator is working as
//       as it should. I get zero often, which ain't so random. So, I'm just using srandom() and random().

#include "replica.h"
#include "stdlib.h"
#include "time.h"

int rng2_get_time_seed () {
  return time(NULL); 
}

void rng2_seed (int n) {
  //if (n < 0 ) srandom(time(NULL)); 
  //else srandom(n);
  srandom(n);
}

double rng2() {
  return random() / (double)RAND_MAX; 
}
