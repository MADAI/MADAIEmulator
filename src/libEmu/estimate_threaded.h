#ifndef __INC_ESTIMATE_THREADED__
#define __INC_ESTIMATE_THREADED__

//#include "main.h"
#include "pthread.h"
#include "gsl/gsl_rng.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"

#include "../main.h"
/* common data block for most options to be passed around */
#include "../optstruct.h"
#include "../modelstruct.h"


//! used to pass the args into estimate_thread_function
struct estimate_thetas_params{
	// these are direct copies of the respective data structures
	optstruct* options;
	modelstruct* the_model;
	// things needed specifically for the estimation
	gsl_rng* random_number;	
	gsl_matrix* h_matrix;
	int max_tries;
} estimate_thetas_params;

// have to include this after derining the above struct or you get tied in knots
#include "maxlbfgs.h"

#define USEMUTEX

void estimate_thetas_threaded(modelstruct* the_model, optstruct* options);

// this won't work unless it has access to the nasty globals in estimate_threaded.c
void* estimate_thread_function(void* args);
int get_number_cpus(void);


void setup_params(struct estimate_thetas_params *params, modelstruct* the_model, optstruct* options, int nthreads, int max_tries);



// see the source for defs of the number of threads etc etc
#endif

