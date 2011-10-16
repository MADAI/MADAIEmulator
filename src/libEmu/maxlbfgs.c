#include "maxlbfgs.h"
#include "gsl/gsl_errno.h"

#define SCREWUPVALUE -20000

/**
 * @file 
 * @author Chris Coleman-Smith cec24@phy.duke.edu
 * @version 1.1
 * @section DESCRIPTION
 * 
 * contains routines to use the lbfgs maximisation method
 * this is a constrained modified-newtons method, see the 
 * file lbfgs/routines.f and the files lbfgs/drivers[1-3].f for 
 * more information about how it works and what it does 
 *
 * the maxWithBLAH interface of rand, max_tries etc has become kind of standard, see the neldermead, and 
 * plain bfgs methods.
 *  
 * April '10 updated with support for linear regression model
 */
void maxWithLBFGS(struct estimate_thetas_params *params){
									
	/* 
	 * this is the main routine for driving the lbfgs method
	 */

	int tries = 0;
	double likelyHood = 0.0;
	double bestLikelyHood = SCREWUPVALUE;
	gsl_vector *xInit = gsl_vector_alloc(params->options->nthetas);
	gsl_vector *xFinal = gsl_vector_alloc(params->options->nthetas);
	gsl_vector *xBest = gsl_vector_alloc(params->options->nthetas);
	double *tempVec = malloc(sizeof(double)*params->options->nthetas);
	pthread_t self = pthread_self(); // this is your thread id

	params->h_matrix = gsl_matrix_alloc(params->options->nmodel_points, params->options->nregression_fns);

	makeHMatrix(params->h_matrix, params->the_model->xmodel,params->options->nmodel_points, params->options->nparams, params->options->nregression_fns);
	
	gsl_vector_set_zero(xBest);
	gsl_vector_set_zero(xFinal);
	set_random_initial_value(params->random_number, xInit, params->options->grad_ranges, params->options->nthetas);
	
	/* printf("max_tries = %d\n", params->max_tries); */
	while(tries < params->max_tries) {
		// ccs, shouldn't we be doing getGradientExactGauss here?
		// holy-craperal! We are doing the exact gradient, but we still pass in this
		// shitty pointer too. whyyyy? (changed to null)
		doBoundedBFGS(&evalFnLBFGS, &getGradientExact, params->options->grad_ranges, xInit, xFinal, params->options->nthetas, 500, (void*)params);
		
		copy_gslvec_vec(xFinal, tempVec, params->options->nthetas);
		likelyHood = evalFnLBFGS(tempVec, params->options->nthetas, (void*)params);
		
		/*annoying!
		 *fprintPt(stdout, self);
		 *printf(":L = %g:try = %d\n", likelyHood, tries);
		 */
		
		if(likelyHood > bestLikelyHood && (isnan(likelyHood) == 0 && isinf(likelyHood) == 0)){
			bestLikelyHood = likelyHood;
			gsl_vector_memcpy(xBest, xFinal);

			/* fprintPt(stdout, self); */
			/* printf(":best = %g\n", bestLikelyHood); */
		}
		tries++;
		set_random_initial_value(params->random_number, xInit, params->options->grad_ranges, params->options->nthetas);
		gsl_vector_set_zero(xFinal);
	}
	
	//printf("Final Best = %g\n", bestLikleyHood);
	if(bestLikelyHood == SCREWUPVALUE) {
		fprintf(stderr, "maximisation didn't work at all, relax your ranges\n");
	}

	gsl_vector_memcpy(params->the_model->thetas, xBest);
	
	gsl_vector_free(xInit);
	gsl_vector_free(xFinal);
	gsl_vector_free(xBest);
	free(tempVec);
}


/**
 * returns the loglikleyhood for the final step of estimate_thread_function
 *
 * used in estimate_thread_function to set "my_theta_val"
 * this is a wrapper for evalFnLBFGS
 * 
 * see estimate_threaded.h for a spec of estimate_thetas_params
 */
double evalLikelyhoodLBFGS_struct(struct estimate_thetas_params *params){
	int i;
	double *xinput = MallocChecked(sizeof(double)*params->options->nthetas);
	double likelihood = 0.0;

	for(i = 0; i < params->options->nthetas; i++) 	/* setup xinput*/
		xinput[i] = gsl_vector_get(params->the_model->thetas, i);
	
	likelihood = evalFnLBFGS(xinput, params->options->nthetas, params);

	free(xinput);
	return(likelihood);
}


	
/**
 * Calculates the loglikelihood for a given set of thetas
 * 
 * @parameter xinput -> a flat double vector of the position in parameter space to be evaluated
 * @parameter nthetas -> length of xinuput, the number of hyperparams in the statistical model
 * @parameter args -> a voided (struct evalFnLBFGSArgs)
 * @return the loglikleyhood of xinput
 */
double evalFnLBFGS(double *xinput, int nthetas, void* args){
	struct estimate_thetas_params *params = (struct estimate_thetas_params*) args;

	// need to disable the gsl error handler
	gsl_error_handler_t *temp_handler;
	gsl_matrix* covariance_matrix = gsl_matrix_alloc(params->options->nmodel_points, params->options->nmodel_points);
	gsl_matrix* cinverse = gsl_matrix_alloc(params->options->nmodel_points, params->options->nmodel_points);
	gsl_matrix* temp_matrix = gsl_matrix_alloc(params->options->nmodel_points, params->options->nmodel_points);

	gsl_vector *xk = gsl_vector_alloc(nthetas);
	//gsl_permutation *c_LU_permutation = gsl_permutation_alloc(params->options->nmodel_points);	
	double determinant_c = 0.0;
	double temp_val = 0.0;
	int i, j;
	int cholesky_test = 0;
		
	// copy the given double vector into the gsl vector
	copy_vec_gslvec(xinput, xk, params->options->nthetas );

	// make the covariance matrix 
	// using the random initial conditions! (xold not thetas)
	makeCovMatrix(covariance_matrix, params->the_model->xmodel, xk, params->options->nmodel_points, nthetas, params->options->nparams);
	gsl_matrix_memcpy(temp_matrix, covariance_matrix);
	
	//print_matrix(temp_matrix, params->options->nmodel_points, params->options->nmodel_points);

	// do a cholesky decomp of the cov matrix, LU is not stable for ill conditioned matrices
	temp_handler = gsl_set_error_handler_off();
	cholesky_test = gsl_linalg_cholesky_decomp(temp_matrix);

	if(cholesky_test == GSL_EDOM){
		FILE *fptr;
		fprintf(stderr, "trying to cholesky a non postive def matrix, sorry...\n");
		fprintf(stderr, "matrix dumped to chol-err.dat\n");
		fptr = fopen("chol-err.dat", "w");
		for(i = 0; i < params->options->nmodel_points; ++i){
			for(j = 0; j < params->options->nmodel_points; ++j){
				fprintf(fptr, "%lf", gsl_matrix_get(temp_matrix, i, j));				
			}
			fprintf(fptr, "\n");
		}
		fclose(fptr);
		exit(1);
	}
	gsl_set_error_handler(temp_handler);

	// find the determinant and then invert 
	// the determinant is just the trace squared
	determinant_c = 1.0;
	for(i = 0; i < params->options->nmodel_points; i++)
		determinant_c *= gsl_matrix_get(temp_matrix, i, i);
	determinant_c = determinant_c * determinant_c;

	//printf("det CHOL:%g\n", determinant_c);	
	gsl_linalg_cholesky_invert(temp_matrix);
	gsl_matrix_memcpy(cinverse, temp_matrix);

	// temp_val is now the likelyhood for this answer
	temp_val = getLogLikelyhood(cinverse, determinant_c, params->the_model->xmodel, params->the_model->training_vector, xk, params->h_matrix, params->options->nmodel_points, params->options->nthetas, params->options->nparams, params->options->nregression_fns);
		
	/* fprintf(stderr,"L:%f\n", temp_val); */
	/* print_vector_quiet(xk, nthetas); */
		
	gsl_matrix_free(covariance_matrix);
	gsl_matrix_free(cinverse);
	gsl_matrix_free(temp_matrix);
	gsl_vector_free(xk);
	return(temp_val);
}

/**
 * this gets the exact gradient, xk are the thetas we are interested in right now
 *
 *
 * the gradient is a vector in each of the theta directions
 * theta_0 (the scale of the cov fn)
 * theta_1 (the nugget)
 * theta_2..theta_n (the directions of the cov-fn)
 *
 * the gradient in the length directions (theta_3 etc) is computed exactly, by functions 
 * named derivative_l_<cov_fn_name> in emulator.c. The function pointer  maxlbfgs.h:makeGradMatLength is 
 * set to the correct derivative function by a call to optstruct.c:setup_cov_fn
 * 
 * 
 * dL/dTheta = -1/2 tr (Cn^{-1} dCn/dtheta) + 1/2 tn^{t}Cn^{-1} dCn/dtheta Cn^{-1} tn
 *  where t_n are the n training points
 * 
 */
void getGradientExact(double *xinput, double* gradient, int nparamsEstimate, void* args){
	int nmpoints, nthetas, i, j;
	int nparams;
	double amp; 

	struct estimate_thetas_params *params = (struct estimate_thetas_params*)args;
	nmpoints = params->options->nmodel_points;
	nthetas = nparamsEstimate;
	nparams = params->options->nparams;
	int cholesky_test;
	gsl_error_handler_t *temp_handler;
	gsl_matrix* covariance_matrix = gsl_matrix_alloc(nmpoints, nmpoints);
	gsl_matrix* cinverse = gsl_matrix_alloc(nmpoints, nmpoints);
	gsl_matrix* temp_matrix = gsl_matrix_alloc(nmpoints, nmpoints);
	


	gsl_vector *xk = gsl_vector_alloc(nthetas);

	copy_vec_gslvec(xinput, xk, nthetas);

	// using the random initial conditions! (xold not thetas)
	makeCovMatrix(covariance_matrix, params->the_model->xmodel, xk, nmpoints, nthetas, nparams);
	gsl_matrix_memcpy(temp_matrix, covariance_matrix);

	//print_matrix(temp_matrix, nmpoints, nmpoints);

	// do a cholesky decomp of the cov matrix, LU is not stable for ill conditioned matrices
	temp_handler = gsl_set_error_handler_off();
	cholesky_test = gsl_linalg_cholesky_decomp(temp_matrix);
	if(cholesky_test == GSL_EDOM){
		FILE *fptr;
		fprintf(stderr, "trying to cholesky a non postive def matrix, sorry...\n");
		fprintf(stderr, "matrix dumped to chol-err.dat\n");
		fptr = fopen("chol-err.dat", "w");
		for (i = 0; i < params->options->nmodel_points; ++i){
			for (j = 0; j < params->options->nmodel_points; ++j){
				fprintf(fptr, "%lf", gsl_matrix_get(temp_matrix, i, j));				
				}
			fprintf(fptr, "\n");
		}

		exit(1);
	}
	gsl_set_error_handler(temp_handler);

	gsl_linalg_cholesky_invert(temp_matrix);
	gsl_matrix_memcpy(cinverse, temp_matrix);



	// C(x,y) = theta_0 c(x,y) + theta_1
	// make the first component, here dC/dN_0 = 1/theta0 * C
	// the second component, dC/dN_1 = 1_delta(ij) (easssy)
	// the rest: dC/dN_{h} = amp * dC/d\theta
	
	// first, get the scale parameter

	amp = gsl_vector_get(xk, 0); 

	gsl_matrix_memcpy(temp_matrix, covariance_matrix);

	gsl_matrix_scale(temp_matrix, 1.0/amp); //1/theta1* C
	gradient[0] = -1.0*getGradientCn(temp_matrix, cinverse, params->the_model->training_vector, nmpoints,  nthetas);

	// next, the nugget
	gsl_matrix_set_identity(temp_matrix);
	gradient[1] = -1.0*getGradientCn(temp_matrix, cinverse, params->the_model->training_vector, nmpoints,  nthetas);
	
	for(i = 2; i < nthetas; i++){
		// this should be called with xmodel not the cov matrix?
		makeGradMatLength(temp_matrix, params->the_model->xmodel, gsl_vector_get(xk, i) , i, nmpoints, nparams);
		gradient[i] = -1.0*amp*getGradientCn(temp_matrix, cinverse, params->the_model->training_vector, nmpoints,  nthetas);
	}
	
	gsl_vector_free(xk);
	gsl_matrix_free(covariance_matrix);
	gsl_matrix_free(cinverse);
	gsl_matrix_free(temp_matrix);
}

/**
 * 
 * computes the gradient for a given hyper-parameter direction, the important arguments here are:
 * we rely on the CONVENTION that all cov-fns are laid out as follows:
 * C(x,y) = theta_0 * c(x,y, theta_2,theta_3,...) + theta_1 
 * i.e each cov fn has a scale given by the first hyper param and a nugget term given by the 
 * second. 
 * this is important when we compute the gradient
 * 
 * 
 * @param dCdtheta a matrix of the covariance function at each point differentiated wrt to the theta 
 * that we want the gradient in.
 * @param cinverse the inverted cov matrix
 * @return the value of the gradient in this direction
 * 
 * grad = -1/2*trace( cinverse %*% dCdtheta ) + 1/2 tn^{t} %*% cinverse %*% dCdtheta %*% tn 
 *
 * rays of light shone down on me, and all my sins, were pardoned
 */
double getGradientCn(gsl_matrix * dCdtheta, gsl_matrix *cinverse,  gsl_vector* training_vector,int nmodel_points, int nthetas){
	double grad = 0;
	double trace = 0;
	int i;
	// this is going to be Cn^{-1}*dCn/dTheta
	gsl_matrix *temp = gsl_matrix_calloc(nmodel_points, nmodel_points);
	// this is Cn{-1} * training
	gsl_vector *v = gsl_vector_alloc(nmodel_points);
	gsl_vector *w = gsl_vector_alloc(nmodel_points);
	/* — Function: int gsl_blas_dsymm (CBLAS_SIDE_t Side, CBLAS_UPLO_t Uplo, double alpha, const gsl_matrix * A, const gsl_matrix * B, double beta, gsl_matrix * C) */
	//gsl_blas_dsymm(CblasLeft, CblasUpper, 1.0, cinverse, dCdtheta, 0.0, temp);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, cinverse, dCdtheta, 0.0, temp);

	
	for(i = 0; i < nmodel_points; i++)
		trace += gsl_matrix_get(temp, i, i);
	trace *= -(0.5);
	
	//printf("trace = %lf\n", trace);

	/* — Function: int gsl_blas_dsymv (CBLAS_UPLO_t Uplo, double alpha, const gsl_matrix * A, const gsl_vector * x, double beta, gsl_vector * y) */
	//gsl_blas_dsymv(CblasUpper, 0.5, cinverse, training_vector, 0.0, v); // v = cinverse %*% training_vector
	gsl_blas_dgemv(CblasNoTrans, 0.5, cinverse, training_vector, 0.0, v);
	//gsl_blas_dsymv(CblasUpper, 1.0, temp, v, 0.0, w); // w = (cinverse %*% dCdthtea) %*% v
	gsl_blas_dgemv(CblasNoTrans, 1.0, temp, v, 0.0, w); // w = (cinverse %*% dCdthtea) %*% v
	
	gsl_blas_ddot(training_vector, w, &grad);

	//printf("grad = %lf\n", grad);
	
	grad += trace;

	gsl_matrix_free(temp);
	gsl_vector_free(v);
	gsl_vector_free(w);
	return(grad);
}
	


/**
 * init the vector x to a set of random values which are sampled from a 
 * uniform dist (from rand) generated on the ranges given by the 
 * matrix ranges
 */
void set_random_initial_value(gsl_rng* rand, gsl_vector* x, gsl_matrix* ranges,int  nthetas){
	int i;
	double range_min; 
	double range_max;
	double the_value;
	
	for(i = 0; i < nthetas; i++){
		range_min = gsl_matrix_get(ranges, i, 0);
		range_max = gsl_matrix_get(ranges, i, 1);
		// set the input vector to a random value in the range
		the_value = gsl_rng_uniform(rand) * (range_max - range_min) + range_min;
		//printf("theta %d set to %g\n", i, the_value);
		//gsl_vector_set(x, gsl_rng_uniform(rand)*(range_max-range_min)+range_min, i);
		gsl_vector_set(x, i, the_value);
	}
}
