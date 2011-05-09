#ifndef _INC_RESULTSTRUCT_
#define _INC_RESULTSTRUCT_

#include "optstruct.h"
#include "modelstruct.h"
#include "libEmu/emulator.h" // because print_matrix lives here?
#include "useful.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"


/**
 * \file resultstruct.h
 * \brief defines the resultstruct which holds emulated values
 */

/**
 * \struct resultstruct
 * \brief holds the emulated values of the gp emulated model
 */
typedef struct resultstruct{
	/** the points that the emluator has been evaluated at*/
	gsl_matrix* new_x; 
	/** the emulated mean at these points */
	gsl_vector* emulated_mean;
	/** the emulated variance at these points */
	gsl_vector* emulated_var;
	/** the optstruct which "owns" these results */ 
	optstruct* options;
	/** the modelstruct which led to these results */
	modelstruct* model;
} resultstruct;

void free_resultstruct(resultstruct *res);
void alloc_resultstruct(resultstruct *res, optstruct *opts);
void copy_resultstruct(resultstruct *dst, resultstruct *src);
void fill_resultstruct(resultstruct* res, optstruct* options, char** input_data, int number_lines);


#endif
