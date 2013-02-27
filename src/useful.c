/**
useful.c

Copyright 2009-2013, Duke University and The University of North
Carolina at Chapel Hill.

ACKNOWLEDGMENTS:
  This software was written in 2009-2013 by Hal Canary
  <cs.unc.edu/~hal>, and Christopher Coleman-Smith
  <cec24@phy.duke.edu> while working for the MADAI project
  <http://madai.us/>.

LICENSE:
  (Since we link to the GNU Scientific Library, we MUST use GPL.)
  This program is free software: you can redistribute it and/or modify
  it under the terms of version 3 of the GNU General Public License as
  published by the Free Software Foundation.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  A copy of version 3 of the GNU General Public License can be found at
  <http://www.gnu.org/licenses/gpl-3.0-standalone.html>.
*/

// some useful things
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "sys/time.h"
#include "useful.h"
#include "assert.h"



//! checks for null
void *MallocChecked(size_t size){
  void *r = malloc(size);
  if( r == NULL)
    perror("malloc");
  return(r);
}

//! print a vector to stderr
void print_vector_quiet(gsl_vector *x, int n){
int i;
  for(i =0; i < n; i++){
    fprintf(stderr, "%g\t", gsl_vector_get(x, i));
  }
  fprintf(stderr,"\n");
}

// RNG
// tries to read from /dev/random, or otherwise uses the system time
unsigned long int get_seed(void){
  unsigned int seed;
  struct timeval tv;
  FILE *devrandom;

  if((devrandom = fopen("/dev/random", "r")) == NULL){
    gettimeofday(&tv, 0);
    seed = tv.tv_sec + tv.tv_usec;
    //fprintf(stderr,"Got seed %u from gettimeofday()\n", seed);
  }
  else {
    int ret = fread(&seed, sizeof(seed), 1, devrandom);
    assert(ret == 1);
    //fprintf(stderr, "Got seed %u from /dev/random\n", seed);
    fclose(devrandom);
  }
  return(seed);
}

// RNG
// tries to read from /dev/random, or otherwise uses the system time
unsigned long int get_seed_noblock(void){
  unsigned long int seed;
  struct timeval tv;
  FILE *devrandom;

  if((devrandom = fopen("/dev/urandom", "r")) == NULL){
    gettimeofday(&tv, 0);
    seed = tv.tv_sec + tv.tv_usec;
    //fprintf(stderr,"Got seed %u from gettimeofday()\n", seed);
  }
  else {
    int ret = fread(&seed, sizeof(seed), 1, devrandom);
    assert (ret == 1);
    //fprintf(stderr, "Got seed %u from /dev/random\n", seed);
    fclose(devrandom);
  }
  return(seed);
}

int read_integer(FILE *fptr) {
  assert(fptr != NULL);
  int i;
  int ret = fscanf(fptr, "%d%*c", &i);
  assert(ret == 1);
  return i;
}
double read_double(FILE *fptr) {
  assert(fptr != NULL);
  double d;
  int ret = fscanf(fptr, "%lf%*c", &d);
  assert(ret == 1);
  return d;
}

char * read_word(FILE *fptr) {
  assert(fptr != NULL);
  char buffer[4096]; // more than we need on the stack.
  int ret = fscanf(fptr, "%4095s", buffer);
  assert(ret == 1);
  buffer[4095] = '\0'; // make sure string is terminated.
  char * string = malloc(strlen(buffer) + 1);
  strcpy (string, buffer);
  return string;
}

int check_word_is(FILE *fptr, const char * str) {
  assert(fptr != NULL);
  assert(str != NULL);
  char buffer[4096];
  int ret = fscanf(fptr, "%4095s", buffer);
  assert(ret == 1);
  buffer[4095] = '\0';
  return str_equal(buffer, str);
}

char ** allocate_string_array(int size) {
  assert(size >= 0);
  char ** r = (char **)malloc((size + 1) * sizeof(char*));
  if (r == NULL) {
    perror("malloc");
		return NULL;
	}
  r[size] = NULL;
  return r;
}

void free_string_array(char ** string_array) {
  if (string_array == NULL)
    return;
  char ** ptr = string_array;
  while (*ptr != NULL)
    free(*(ptr++));
  free(string_array);
}

void discard_line(FILE *fptr) {
   int ch;
   while ((EOF != (ch = getc(fptr))) && ('\n' != ch));
}

void discard_comments( FILE * fptr, char comment_character ) {
  int c = getc( fptr );
  if ( (c == EOF) || ferror( fptr ) )
    return;
  while ( c == comment_character ) {
    discard_line(fptr);
    c = getc(fptr);
  }
  if ( EOF == ungetc(c, fptr) ) {
    return;
  }
  return;
}

