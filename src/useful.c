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
		perror("memory wasn't allocated");
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
