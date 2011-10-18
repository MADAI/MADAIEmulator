#ifndef __INC_OPTSTRUCT__
#define __INC_OPTSTRUCT__

#include <gsl/gsl_matrix.h>


/**
 * @file optstruct.h
 * \brief defines the optstruct which holds all the mundane options and dimensions etc
 */

#define POWEREXPCOVFN 1
#define MATERN32 2
#define MATERN52 3


/**
 * \struct optstruct
 * \brief holds the main parameters etc needed by all the emulator fns
 * 
 * this is the most important structure after the modelstruct, this contains
 * all the dimensions of xmodel, training_vec etc
 * 
 */
typedef struct optstruct{
	/**
	 * the number of parameters in the covariance fn
	 */
	int nthetas;
	/**
	 * the number of parameters in the model to be emulated, this sets the dimensionality
	 * of the estimation space and the "hardness" of the estimation process
	 */
	int nparams;
	/**
	 * the number of locations in the model space that we have training values at
	 * another contributing factor to how complex the problem becomes, the covariance matrix
	 * scales as (nmodel_points**2) and we need to invert this every step of the estimation 
	 * process
	 */
	int nmodel_points;
	/**
	 * how many points to evaluate the emulated model at
	 */
	int nemulate_points;

	/**
	 * what order (if any) should the regression model be
	 * 0 -> a single constant value (a0) 
	 * 1 -> a constant plus a slope vector (a0 + a1*x)
	 * 2 -> (a0 + a1*x + a2*x^2)
	 * 3 -> (a0 + a1*x + a2*x^2 + a3*x^3)
	 * >3 (not supported) 
	 */
	int regression_order;

	/**
	 * indirectly controls the shape of the regression model, be very careful with this one
	 * this is set correctly by calling setup_regression
	 */
	int nregression_fns;


	/** 
	 * fixed nugget?
	 * allow the user to try and supply a nugget which will not be optimized over, but fixed 
	 */
	int fixed_nugget_mode;
	double fixed_nugget;

	
	/** 
	 * set which cov fn to use, can be one of 
	 * POWEREXPCOVFN
	 * MATERN32
	 * MATERN52
	 * 
	 * this is determined by setup_cov_fn
	 */
	int cov_fn_index; 
	
	// set this to not zero if you want to use the length scales set by the data
	int use_data_scales;

	/** this holds the ranges for the optimisation routine*/
	gsl_matrix* grad_ranges;

} optstruct;

void free_optstruct(optstruct *opts);
void copy_optstruct(optstruct *dst, optstruct* src);
void dump_optstruct(FILE *fptr, optstruct* opts);
void load_optstruct(FILE *fptr, optstruct* opts);

void setup_cov_fn(optstruct *opts);
void setup_regression(optstruct *opts);


#include "modelstruct.h"

void setup_optimization_ranges(optstruct* options, modelstruct* the_model);



#endif
