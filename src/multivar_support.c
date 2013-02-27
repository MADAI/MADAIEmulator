#include "multivar_support.h"
#include "multi_modelstruct.h"
#include "emulator_struct.h"
#include "modelstruct.h"
#include "libEmu/estimate_threaded.h"
#include <math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>



/**
 * estimates the thetas for an allocated multvariate model and outputs the 
 * info to outfp
 * 
 * requries: 
 * m must have been succesfully allocated and pca'd
 * outfp is an open fileptr
 */
void estimate_multi(multi_modelstruct *m, FILE* outfp)
{
	int i;
	for(i = 0; i < m->nr; i++){
		estimate_thetas_threaded(m->pca_model_array[i], m->pca_model_array[i]->options);
	}
	// dump the trained modelstruct, this maybe should go somewhere else
	dump_multi_modelstruct(outfp, m);
}

multi_emulator *alloc_multi_emulator(multi_modelstruct *m)
{
	int i;
	multi_emulator *e = (multi_emulator*)malloc(sizeof(multi_emulator));
	e->nt = m->nt;
	e->nr = m->nr;
	e->nparams = m->nparams;
	e->nmodel_points = m->nmodel_points;
	// have to read these out of the pca array
	e->nregression_fns = m->pca_model_array[0]->options->nregression_fns;
	e->nthetas = m->pca_model_array[0]->options->nthetas;
	e->model = m;
	
	// allocate the nr emulators we need
	e->emu_struct_array = (emulator_struct**)malloc(sizeof(emulator_struct*)*e->nr);

	for(i = 0; i < e->nr; i++)
		e->emu_struct_array[i] = alloc_emulator_struct(m->pca_model_array[i]);
	
	return e;
}

/**
 * free allocated memory
 */
void free_multi_emulator(multi_emulator *e)
{
	int i;
	free_multimodelstruct(e->model);
	for(i = 0; i < e->nr; i++)
		free_emulator_struct(e->emu_struct_array[i]);
	free(e->emu_struct_array);
}

/**
 * sample the multivariate emulator at the_point, returns values in the *PCA* space. 
 * The returned values will be vectors of length emu->model->nr *not* emu->model->nt
 *
 * requires:
 * emu has been allocated from a multi_modelstruct, and 
 * the_mean and the_variance point to allocated gsl_vectors of length (emu->model->nr)
 *
 * @argument emu: a multi_emulator structure created from an estimated multi_modelstruct with alloc_multi_emulator
 * @argument the_point: the location in parameter space (nparams) length vector
 * @argument the_mean: the emulated mean values (nr length)
 * @argument the_variance: the emulated variance values (nr length)
 */

void emulate_point_multi_pca(multi_emulator *emu, gsl_vector *the_point,
												 gsl_vector *the_mean, gsl_vector *the_variance)
{
	int i;
	// sample the nr emulators
	// the results are directly fed back  into the_mean and the_variance
	for(i = 0; i < emu->nr; i++)
		emulate_point(emu->emu_struct_array[i], the_point, gsl_vector_ptr(the_mean, i), gsl_vector_ptr(the_variance, i));
}



/**
 * emulate the values of a point in a multivariate model, returns
 * values in the REAL Observable space.
 *
 * This means that the values output here will be in the same space as
 * the training data.
 * 
 * requries:
 * emu has been allocated from a multi_modelstruct
 * the_mean points to allocated gsl_vector of length (emu->model->nt)
 * the_covariance points to allocated gsl_matrix of size (emu->model->nt,emu->model->nt)
 * 
 * @argument emu: a multi_emulator structure created from an estimated
 *                multi_modelstruct with alloc_multi_emulator
 * @argument the_point: the location in parameter space (nparams) length vector
 * @argument the_mean: the emulated mean values (t length)
 * @argument the_covariance: the emulated covariance matrix (txt matrix)
 */
void emulate_point_multi(multi_emulator *emu, gsl_vector *the_point,
												 gsl_vector *the_mean, gsl_matrix *the_covariance)
{
	int i, j;
	int nr = emu->nr;
	int nt = emu->nt;
	double vec_mat_sum;
	// the mean and variance in the PCA space
	gsl_vector *mean_pca = gsl_vector_alloc(nr);
	gsl_vector *var_pca = gsl_vector_alloc(nr); 

	// first we sample the nr emulators
	for(i = 0; i < emu->nr; i++)
		emulate_point(emu->emu_struct_array[i], the_point, gsl_vector_ptr(mean_pca, i), gsl_vector_ptr(var_pca, i));

	/**
	 *  now we have to project back into the REAL space from the PCA space
	 * mean_real (nt) = training_mean (nt) + pca_evec_r %*% diag(sqrt(pca_evals_R)) %*% mean_pca
	 */
	gsl_vector *mean_real = gsl_vector_alloc(nt);
	gsl_vector *temp = gsl_vector_alloc(nt);
	gsl_vector_memcpy(mean_real, emu->model->training_mean);

	vec_mat_sum = 0.0;
	for(i = 0; i < nt; i++){
		for(j = 0; j < nr; j++)
			vec_mat_sum += gsl_matrix_get(emu->model->pca_evecs_r, i, j) * sqrt(gsl_vector_get(emu->model->pca_evals_r, j))*gsl_vector_get(mean_pca, j);
		// save the sum scaled by the sqrt of the eval
		//gsl_vector_set(temp, i, vec_mat_sum*sqrt(gsl_vector_get(emu->model->pca_evals_r, i)));
		gsl_vector_set(temp, i, vec_mat_sum);
		vec_mat_sum = 0.0;
	}
	
	gsl_vector_add(mean_real, temp);
	gsl_vector_memcpy(the_mean, mean_real); // save the final mean

	/**
	 * project back the variance 
	 * the_covariance = pca_evecs * diag(pca_evals) * diag(var_pca) * pca_evecs.Transpose
	 */
	int k;
	gsl_vector_set_zero(temp);
	/* for (k = 0; k < nr; k++) { */
	/* 	printf("var[%d] = %.17f\n",k,gsl_vector_get(var_pca, k)); */
	/* } */
	for (i = 0; i < nt; i++) {
		for (j = 0; j < nt; j++) {
 			vec_mat_sum = 0.0;
			for (k = 0; k < nr; k++) {
				vec_mat_sum +=
					( gsl_matrix_get(emu->model->pca_evecs_r, i, k) * 
					  gsl_vector_get(emu->model->pca_evals_r, k) * 
					  gsl_vector_get(var_pca, k) * 
					  gsl_matrix_get(emu->model->pca_evecs_r, j,k) );
			}
			gsl_matrix_set(the_covariance, i, j, vec_mat_sum);
		}
	}

	gsl_vector_free(mean_real);
	gsl_vector_free(temp);
	gsl_vector_free(mean_pca);
	gsl_vector_free(var_pca);
}


