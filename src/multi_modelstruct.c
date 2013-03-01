/**
multi_modelstruct.c

Copyright 2012-2013, Duke University and The University of North
Carolina at Chapel Hill.

ACKNOWLEDGMENTS:
  This software was written in 2010-2013 by Hal Canary
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

#include "multi_modelstruct.h"
#include "multivar_support.h"
#include "useful.h"

#include <math.h>
#include <assert.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>

/**
 * ccs, 05.06.2012
 * \todo this all needs to be checked carefully against an example that works in R
 * \todo valgrind for mem-leaks
 * \bug no error checking in linalg
 * \bug no error checking in alloc
 * \bug is there error checking in file handles?
 *
 * ccs, 19.06.2012
 * if you allocate a multi_modelstruct m
 * then m->pca_model_array[i]->xmodel == m->xmodel
 * the value in the pca_model_array is just a pointer to m, do we really want this?
 */


/**
 * \brief set the allowed ranges for the maximisation routine lbfgs
 *
 * \todo add a flag to optstruct for log-scaled and regular thetas
 *
 * summer-2011, we should try setting lower-limits on the length-scale thetas
 * interms of the average nearest-neighbour distance in the input data.
 * i.e in a 1d model we may have samples
 *     x --- x --- x ---  x
 * it doesn't make sense to have our length scale theta_1 < ---
 * because we dont have information at that frequency!
 *
 * this fills the options->grad_ranges matrix with lower and upper bounds to be used in the
 * bounded bfgs maximisation routine lbfgs (see libEmu/maxlbfgs.c) for more info on this.
 *
 *
 * this is not the case for the other fns, but the rest of the code may have the assumption
 * frozen into it that the ranges *are* log-scaled.
 *
 * the gsl_matrix* that is returned needs to be gsl_matrix_free()ed afterwards.
 */
static gsl_matrix * optimization_ranges(
	gsl_vector *sample_scales,
	int nthetas,
	int cov_fn_index,
	int use_data_scales,
	int fixed_nugget_mode,
	double fixed_nugget)
{
	int i = 0;

	/** does it make sense to have the upper limit on theta be E(10.) = 22026? probably not
	 */
	double bigRANGE = 10.0;
	double rangeMin = 0.0, rangeMax = 0.0;
	double fixedNuggetLeeWay = 0.0 ;
	double rangeMinLog = 0.0001;

	double rangeMinNugget = -5.0;//log(0.0011);
	double rangeMaxNugget = -2.0;//log(0.01); //what's a sensible upper limit here?
	/**
	 * alloc the grad_ranges matrix in the options and
	 * put in some sensible defaults
	 */
	gsl_matrix * grad_ranges = gsl_matrix_alloc(nthetas, 2);

	/*
	 * setup that we're using a log scale
	 */
	if(cov_fn_index == POWEREXPCOVFN){
		rangeMin = rangeMinLog;
		rangeMax = 5; // exp(5) = 148 this is big!
	} else {
		rangeMin = 0;
		rangeMax = bigRANGE;
	}

	gsl_matrix_set(grad_ranges, 0, 0, rangeMinLog);
	gsl_matrix_set(grad_ranges, 0, 1, rangeMax);

	gsl_matrix_set(grad_ranges, 1, 0, rangeMinNugget);
	gsl_matrix_set(grad_ranges, 1, 1, rangeMaxNugget);

	if(use_data_scales){
		// use length scales set by data the
		for(i = 2; i < nthetas; i++){

			if(cov_fn_index == POWEREXPCOVFN){
				rangeMin = 0.5*log(gsl_vector_get(sample_scales, i-2));
				// try stopping the max range at 25 x the lower limit...
				rangeMax = log(25*exp(rangeMin));

				} else {
				rangeMin = 0.5*(gsl_vector_get(sample_scales, i-2));
				// try stopping the max range at 10 x the nyquist limit...
				//rangeMax = *(gsl_vector_get(sample_scales, i-2));
			}

			if(rangeMin > rangeMax){
				fprintf(stderr, "#ranges failed\n");
				printf("# %d ranges: %lf %lf\n", i, rangeMin, rangeMax);
				printf("# sampleScale: %lf\n", gsl_vector_get(sample_scales, i-2));
				exit(EXIT_FAILURE);
			}
			gsl_matrix_set(grad_ranges, i, 0, rangeMin);
			gsl_matrix_set(grad_ranges, i, 1, rangeMax);
		}

		if(isinf(rangeMin) == -1){
			rangeMin = 0.00001;
		}

	} else { // use some default scales

		for(i = 2; i < nthetas; i++){
			gsl_matrix_set(grad_ranges, i, 0, rangeMin);
			gsl_matrix_set(grad_ranges, i, 1, rangeMax);
		}

	}

	if(fixed_nugget_mode == 1){
		// force the nugget to be fixed_nugget +- 20%
		fixedNuggetLeeWay = 0.20*(fixed_nugget);
		//gsl_matrix_set(grad_ranges, 1, 0, fixed_nugget - fixedNuggetLeeWay);
		//fix the min at the usual value
		gsl_matrix_set(grad_ranges, 1, 0, rangeMinNugget);
		gsl_matrix_set(grad_ranges, 1, 1, fixed_nugget + fixedNuggetLeeWay);
		printf("# (reset) %d ranges: %lf %lf (nugget)\n", 1, gsl_matrix_get(grad_ranges, 1,0), gsl_matrix_get(grad_ranges, 1,1));
	}


	/**
	 * debug info
	 * print the ranges, this is annoying
	 */
	#ifdef DEBUGGRADRANGES
	double low, high;
	fprintf(stderr,"# grad ranges (logged):\n");
	for(i = 0; i < nthetas; i++){
		low = gsl_matrix_get(grad_ranges, i, 0);
		high = gsl_matrix_get(grad_ranges, i, 1);
		if(i == 0){
			fprintf(stderr,"# %d ranges: %lf %lf (scale)\n", i, low, high);
		} if (i == 1){
			fprintf(stderr,"# %d ranges: %lf %lf (nugget)\n", i, low, high);
		} else if (i > 1) {
			fprintf(stderr,"# %d ranges: %lf %lf\n", i, low, high);
		}
	}
	#endif
	return grad_ranges;
}


/**
 * allocates a multi_modelstruct, like alloc_modelstruct_2,
 * but for multivariate models with t output values at each location
 *
 * @param model_in: (n x d) matrix of the design
 * @param training_matrix: (n x t) matrix of the values of the training values at each of the n locations
 * @param cov_fn_index:  POWEREXPCOVFN, MATERN32, or MATERN52
 * @param regression_order:  0, 1, 2, or 3
 * @param varfrac: the minimum fractional variance that should be retained during the PCA decomp
 *
 * applies a pca decomp to training_matrix to reduce the dimensionality
 *
 */
multi_modelstruct* alloc_multimodelstruct(gsl_matrix *xmodel_in,
                                         gsl_matrix *training_matrix_in,
                                         int cov_fn_index,
                                         int regression_order, double varfrac)
{
  assert(training_matrix_in->size1 == xmodel_in->size1);
  assert(training_matrix_in->size1 > 0);
  assert(training_matrix_in->size2 > 0);
  assert(xmodel_in->size2 > 0);

	assert((0.0 <= varfrac) && (varfrac <= 1.0));
	assert((0 <= cov_fn_index) && (cov_fn_index <= 3));
	assert((0 <= regression_order) && (regression_order <= 3));

  int i;
  double mean_temp = 0.0;
  int nt = training_matrix_in->size2;
  int nmodel_points = xmodel_in->size1;
  int nparams = xmodel_in->size2;
  gsl_vector_view col_view;


  /* use default if out of range */
  if (regression_order < 0 || regression_order > 3)
    regression_order = 0;

  /* use a sensible default for the variance fraction */
  if(varfrac < 0 || varfrac > 1)
    varfrac = 0.95;

  // this doesn't seem to be allocating correctly, seems to be a broken defn of MallocChecked
  // strangely, code will build in this case...
  //multi_modelstruct * model = (multi_modelstruct*)MallocChecked(sizeof(multi_modelstruct));
  multi_modelstruct * model = (multi_modelstruct*)malloc(sizeof(multi_modelstruct));

  // fill in
  model->nt = nt;
  model->nr = 0; // init at zero
  model->nmodel_points = nmodel_points;
  model->nparams = nparams;
  model->xmodel = xmodel_in;
  model->training_matrix = training_matrix_in;
  model->training_mean = gsl_vector_alloc(nt);
  model->regression_order = regression_order;
  model->cov_fn_index = cov_fn_index;

	model->fixed_nugget_mode = 0; /* only option. */
	model->fixed_nugget = 0.0; /* not used. */
	model->nregression_fns	= 1 + ((model->regression_order) * (model->nparams));
	model->use_data_scales = 1;	 /* set grad ranges using the data scales */

  /* ntheta is a function of cov_fn_index and nparams */
  model->number_thetas = (cov_fn_index == POWEREXPCOVFN) ? (nparams + 2) : 3;

	model->sample_scales = fill_sample_scales_vec(model->xmodel);
	model->grad_ranges = optimization_ranges(model->sample_scales,
		model->number_thetas, model->cov_fn_index, model->use_data_scales,
		model->fixed_nugget_mode, model->fixed_nugget);


  /* fill in the mean vector, should probably sum this more carefully... */
  for(i = 0; i < nt; i++){
    col_view = gsl_matrix_column(model->training_matrix, i);
    mean_temp = vector_elt_sum(&col_view.vector, nmodel_points);
    //printf("%lf\n", (mean_temp/((double)nmodel_points)));
    gsl_vector_set(model->training_mean, i, (mean_temp/((double)nmodel_points)) );
  }

  /* carry out the pca decomp on this model, this is defined in multivar_support for now
   * this will fill in nr, pca_eigenvalues, pca_eigenvectors, pca_evals_r, pca_evecs_r
   *
   * this is making a mess if nt = 1
   */
  gen_pca_decomp(model, varfrac);

  /* fill in pca_model_array */
  gen_pca_model_array(model);

  return model;
}

/**
 * fill in pca_model_array
 * requires:
 * - the pca decomp to have been calculated, so nr and all the pca_... fields are allocated and filled in
 * - m to have been allocated up to: nt, nmodel_points, and xmodel
 *
 * is it possible we could start running out of memory doing all these allocs?
 */
void gen_pca_model_array(multi_modelstruct *m)
{
  int nr = m->nr;
  int i;
  gsl_vector_view col_view;
  gsl_vector* temp_train_vector = gsl_vector_alloc(m->nmodel_points);
  //gsl_matrix* temp_xmodel = gsl_matrix_alloc(m->nmodel_points, m->nparams);

  //gsl_matrix_mempcy(temp_xmodel, m->xmodel);

  // alloc the array of nr model structs
  //m->pca_model_array = (modelstruct**)MallocChecked(sizeof(modelstruct*)*nr);
  m->pca_model_array = (modelstruct**)malloc(sizeof(modelstruct*)*nr);
  // fill in the modelstructs correctly
  for(i = 0; i < nr; i++){
    col_view = gsl_matrix_column(m->pca_zmatrix, i);
    gsl_vector_memcpy(temp_train_vector, &(col_view.vector));

    // this isn't copying in the training vector correctly for somereason
    m->pca_model_array[i] = alloc_modelstruct_2(m->xmodel, temp_train_vector,
                                                m->cov_fn_index, m->regression_order);

    // see if brute forcing it will work
    m->pca_model_array[i]->training_vector = gsl_vector_alloc(m->nmodel_points);
    gsl_vector_memcpy(m->pca_model_array[i]->training_vector, temp_train_vector);
  }
  gsl_vector_free(temp_train_vector);
}


/**
 * carries out a pca decomp on m->training_matrix;
 * setting m->nr, m->pca_eigenvalues, m->pca_eigenvectors and
 * initializing m->pca_model_array
 *
 * the pca decomp is pretty simple:
 * let Y_ij = m->y_training
 * let mu_i = (1/nmodel_points) sum_{j=1}^{nmodel_points} Y_{ij}  // the sample mean
 * let ysub_i = Y_i - rep(mu_i,nmodel_points)  //subtract the sample means from each column
 * let sigma_ij = (1/nmodel_points) ( ysub * t(ysub)) // this is the covariance matrix
 *
 * then all we need to do is compute the eigendecomposition
 *
 * sigma_ij = U^{-1} Lambda U
 *
 * where Lambda is a diagonal matrix of t eigenvalues and U is a t x t matrix with the eigenvectors as columns
 *
 * requires:
 * - 0 < vfrac < 1 (0.95 is a good value)
 * - m to have allocated and filled out, nt, nmodel_points, trianing_matrix, (doesn't need xmodel explicitly)
 */
void gen_pca_decomp(multi_modelstruct *m, double vfrac)
{
  FILE *fptr; // for debug output
  int i,j;
  int nt = m->nt;
  int retval;
  double total_variance = 0.0, frac  = 0.0;
  gsl_matrix *pca_zmatrix_temp;
  gsl_matrix *y_sub_mat = gsl_matrix_alloc(m->nmodel_points, nt);
  gsl_matrix *y_temp_mat = gsl_matrix_alloc(m->nmodel_points, nt);

  gsl_matrix *y_cov_mat = gsl_matrix_alloc(nt, nt);

  gsl_vector *evals_temp = gsl_vector_alloc(nt);
  gsl_matrix *evecs_temp = gsl_matrix_alloc(nt,nt);

  gsl_eigen_symmv_workspace *ework = gsl_eigen_symmv_alloc(nt);

  gsl_vector_view col_view;

  // why is this here?
  gsl_matrix_memcpy(y_sub_mat, m->training_matrix);


  // subtract out the mean
  for(i = 0; i < nt; i++){
    //col_view = gsl_matrix_column(y_sub_mat, i);
		#ifdef DEBUGPCA
    fprintf(stderr,"# y(%d) mean: %lf\n", i, gsl_vector_get(m->training_mean, i));
		#endif
    for(j = 0; j < m->nmodel_points; j++){
      gsl_matrix_set(y_sub_mat, j, i, gsl_matrix_get(y_sub_mat, j, i) - gsl_vector_get(m->training_mean, i));
    }
  }

  // compute the sample-variance, by multiplying y_sub_mat, with itself transposed
  gsl_matrix_memcpy(y_temp_mat, y_sub_mat);
  gsl_matrix_set_zero(y_cov_mat);

  /** — Function: int gsl_blas_dgemm (CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB, double alpha, const gsl_matrix * A, const gsl_matrix * B, double beta, gsl_matrix * C)  (always forget this one)*/
  /* want C (nt x nt ) so we need to do: (nt x nmodel_points) * (nmodel_points x nt) */
  /**
   * this is strange, the upper triangle is not obviously correct
   */
  //retval = gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, y_temp_mat, y_sub_mat, 0.0, y_cov_mat);
  retval = gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, y_sub_mat, y_temp_mat, 0.0, y_cov_mat);
  if(retval){
    printf("# gen_pca_decomp:gsl_blas_dgemm %s\n", gsl_strerror(retval));
    exit(EXIT_FAILURE);
  }


  gsl_matrix_scale(y_cov_mat, (1.0/((double)m->nmodel_points)));

  /**
   * debug output
   */
  #ifdef DEBUGPCA
  fptr = fopen("pca-debug.dat","w");
  fprintf(fptr, "# ycov:\n");
  for(j = 0; j < nt; j++){
    for(i = 0; i < nt; i++)
      fprintf(fptr, "%lf ", gsl_matrix_get(y_cov_mat, i, j));
    fprintf(fptr, "\n");
  }
  #endif

  /** now the eigendecomp
   * y_cov_mat is symmetric and better be real so we can use gsl_eigen_symmv,
   * note that the ?upper? triangle of y_cov_mat is borked during this process
   * also: the evals are not sorted by order, but the evectors are in the order of the evalues.
   * so we need to sort the evalues and the evectors correctly before we can use them
   * using
   * — Function: int gsl_eigen_symmv_sort (gsl_vector * eval, gsl_matrix * evec, gsl_eigen_sort_t sort_type)
   * sort them into descending order (biggest first)
   */
  gsl_eigen_symmv(y_cov_mat, evals_temp, evecs_temp, ework);
  gsl_eigen_symmv_sort(evals_temp, evecs_temp, GSL_EIGEN_SORT_VAL_DESC);
  /**
   * eigenvectors are stored in columns of pca_evecs */
  total_variance = vector_elt_sum(evals_temp, nt);


#ifdef DEBUGPCA
  fprintf(fptr, "# evals:\n");
  for(i = 0; i < nt; i++)
    fprintf(fptr, "%lf %lf\n", gsl_vector_get(evals_temp, i), gsl_vector_get(evals_temp, i) / total_variance);

  fprintf(fptr, "# evecs:\n");
  for(j = 0; j < nt; j++){
    for(i = 0; i < nt; i++)
      fprintf(fptr, "%lf ", gsl_matrix_get(evecs_temp, i, j));
    fprintf(fptr, "\n");
  }

  #endif

  i=0;
  while( frac < vfrac && (i+1) < nt){
    frac = (1.0/total_variance) * vector_elt_sum(evals_temp, i);
    i++;
  }
  m->nr = i;

  if(nt == 1){
    //printf("# 1d case, nr=1\n");
    m->nr = 1;
  }

  m->pca_evals_r = gsl_vector_alloc(m->nr);
  m->pca_evecs_r = gsl_matrix_alloc(m->nt, m->nr);
  // debug...
  #ifdef DEBUGPCA
  fprintf(stderr, "# nr: %d frac: %lf\n", m->nr, frac);
  #endif

  for(i = 0; i < m->nr; i++){
    gsl_vector_set(m->pca_evals_r, i, gsl_vector_get(evals_temp, i));
    col_view = gsl_matrix_column(evecs_temp, i);
    gsl_matrix_set_col(m->pca_evecs_r, i, &col_view.vector);
  }

  // fill in pca_zmatrix
  m->pca_zmatrix = gsl_matrix_alloc(m->nmodel_points, m->nr);
  pca_zmatrix_temp = gsl_matrix_alloc(m->nmodel_points, m->nr);
  // zmat: (nmodel_points x nr) = (nmodel_points x nt) * ( nt x nr )
  /** — Function: int gsl_blas_dgemm (CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB, double alpha, const gsl_matrix * A, const gsl_matrix * B, double beta, gsl_matrix * C)  (always forget this one)*/
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,  y_sub_mat, m->pca_evecs_r, 0.0, m->pca_zmatrix);

  #ifdef DEBUGPCA
  fprintf(fptr, "# evecs R:\n");
  for(i = 0; i < m->nt; i++){
    for(j = 0; j < m->nr; j++)
      fprintf(fptr, "%lf ", gsl_matrix_get(m->pca_evecs_r, i, j));
    fprintf(fptr, "\n");
  }
  #endif

  gsl_matrix_free(y_temp_mat);
  y_temp_mat = gsl_matrix_alloc(m->nr, m->nr);
  for(i = 0; i < m->nr; i++) // scale the diagonal by the evalue */
    gsl_matrix_set(y_temp_mat, i, i, 1.0/(sqrt(gsl_vector_get(m->pca_evals_r, i))));

  //print_matrix(y_temp_mat, m->nr, m->nr);

  // if nr != nt this won't work!
  //gsl_matrix_memcpy(y_sub_mat, m->pca_zmatrix);
  gsl_matrix_memcpy(pca_zmatrix_temp, m->pca_zmatrix);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, pca_zmatrix_temp, y_temp_mat, 0.0, m->pca_zmatrix);

  #ifdef DEBUGPCA
  fprintf(fptr, "# zmat:\n");
  for(i = 0; i < 5; i++){
    for(j = 0; j < m->nr; j++)
      fprintf(fptr, "%lf ", gsl_matrix_get(m->pca_zmatrix, i, j));
    fprintf(fptr, "\n");
  }
  fclose(fptr);
  #endif



  gsl_vector_free(evals_temp);
  gsl_matrix_free(evecs_temp);
  gsl_matrix_free(pca_zmatrix_temp);

  gsl_eigen_symmv_free(ework);
  gsl_matrix_free(y_sub_mat);
  gsl_matrix_free(y_temp_mat);
  gsl_matrix_free(y_cov_mat);
}

/**
 * dump the multimodelstruct to fptr, follows from dump_modelstruct_2
 * We first dump the new fields and then iterate through the nr
 * additional models which are dumped as before so we dump a lot of
 * the same info, but this is probably ok, the advantage is that each
 * section defining a model can be pulled out and worked on
 * separately...
 */
void dump_multi_modelstruct(FILE* fptr, multi_modelstruct *m){
  assert(fptr);
  assert(m);
  int i;

  fprintf(fptr, "#\n#\n# Insert comments here.  They will be ignored.\n#\n#\n");
  fprintf(fptr, "VERSION 1\n");
  fprintf(fptr, "PARAMETERS\n");
  print_int(fptr, m->nparams);
  for(i = 0; i < (m->nparams); i++){
    print_str(fptr, m->parameter_names[i]);
    print_double(fptr, m->parameter_minima[i]);
    print_double(fptr, m->parameter_maxima[i]);
  }
  fprintf(fptr, "OUTPUTS\n");
  print_int(fptr, m->nt);
  for(i = 0; i < (m->nt); i++)
    print_str(fptr, m->output_names[i]);

  print_str(fptr,"EMULATOR");

  print_str(fptr,"NUMBER_PCA_OUTPUTS");
  print_int(fptr, m->nr);

  print_str(fptr,"NUMBER_TRAINING_POINTS");
  print_int(fptr, m->nmodel_points);

  print_str(fptr,"COVARIANCE_FUNCTION_INDEX");
  print_int(fptr, m->cov_fn_index);

  print_str(fptr,"REGRESSION_ORDER");
  print_int(fptr, m->regression_order);

  print_str(fptr,"FIXED_NUGGET_MODE");
  print_str(fptr, int_to_bool(m->fixed_nugget_mode));

  print_str(fptr,"FIXED_NUGGET");
  print_double(fptr, m->fixed_nugget);

  print_str(fptr,"USE_DATA_SCALES");
  print_str(fptr, int_to_bool(m->use_data_scales));

  print_str(fptr,"NUMBER_THETAS");
  print_int(fptr, m->number_thetas);

  print_str(fptr,"NUMBER_REGRESSION_FUNCTIONS");
  print_int(fptr, m->nregression_fns);

  print_str(fptr,"PARAMETER_VALUES");
  gsl_matrix_fprint_block(fptr, m->xmodel);
  //  gsl_matrix_fprintf(fptr, m->xmodel, "%.17f");

  print_str(fptr,"OUTPUT_VALUES");
  gsl_matrix_fprint_block(fptr, m->training_matrix);
  //  gsl_matrix_fprintf(fptr, m->training_matrix, "%.17f");

  print_str(fptr,"OUTPUT_PCA_EIGENVALUES");
  gsl_vector_fprintf(fptr, m->pca_evals_r, "%.17f");

  print_str(fptr,"OUTPUT_PCA_EIGENVECTORS");
  gsl_matrix_fprint_block(fptr, m->pca_evecs_r);
  //gsl_matrix_fprintf(fptr, m->pca_evecs_r, "%.17f");

  print_str(fptr,"PCA_DECOMPOSED_Z_VALUES");
  gsl_matrix_fprint_block(fptr, m->pca_zmatrix);
  //gsl_matrix_fprintf(fptr, m->pca_zmatrix, "%.17f");

  print_str(fptr,"MEANS_OF_TRAINING_POINTS");
  gsl_vector_fprintf(fptr, m->training_mean, "%.17f");

  print_str(fptr,"SAMPLE_SCALES");
  gsl_vector_fprintf(fptr, m->sample_scales, "%.17f");

  print_str(fptr,"GRAD_RANGES");
  gsl_matrix_fprint_block(fptr, m->grad_ranges);
  //gsl_matrix_fprintf(fptr, m->grad_ranges, "%.17f");


  for(i = 0; i < (m->nr); i++){
    dump_modelstruct_3(fptr, m->pca_model_array[i],i);
  }
  fprintf(fptr, "END_OF_FILE\n");
}

/**
 * loads a multivariate modelstructure from fptr
 */
multi_modelstruct *load_multi_modelstruct(FILE* fptr){
  int i;
  int number_of_outputs, number_of_pca_outputs;
  int nparams, nmodel_points;
  int cov_fn_index;
  int regression_order;

  double mean_temp;
  gsl_vector_view col_view;
  char ** parameter_names, ** output_names;
  double * parameter_minima, * parameter_maxima;

  discard_comments(fptr, '#');
  if (! check_word_is(fptr, "VERSION")) return NULL;
  if (read_integer(fptr) != 1) {
    fprintf(stderr, "wrong version!\n");
    return NULL;
  }
  if (! check_word_is(fptr, "PARAMETERS")) return NULL;
  nparams = read_integer(fptr);
  parameter_names = allocate_string_array(nparams);
  parameter_minima = MallocChecked(nparams * sizeof(double));
  parameter_maxima = MallocChecked(nparams * sizeof(double));
  for(i = 0; i < (nparams); i++){
    parameter_names[i] = read_word(fptr);
    parameter_minima[i] = read_double(fptr);
    parameter_maxima[i] = read_double(fptr);
  }

  if (! check_word_is(fptr, "OUTPUTS")) return NULL;
  number_of_outputs = read_integer(fptr);
  output_names = allocate_string_array(number_of_outputs);
  for (i = 0; i < number_of_outputs; ++i) {
    output_names[i] = read_word(fptr);
  }

  if (! check_word_is(fptr, "EMULATOR")) return NULL;

  multi_modelstruct * m
    = (multi_modelstruct*)malloc(sizeof(multi_modelstruct));
  m->nparams = nparams;
  m->nt = number_of_outputs;
  m->parameter_names = parameter_names;
  m->output_names = output_names;
  m->parameter_minima = parameter_minima;
  m->parameter_maxima = parameter_maxima;

  if (! check_word_is(fptr, "NUMBER_PCA_OUTPUTS")) return NULL;
  m->nr = number_of_pca_outputs = read_integer(fptr);

  if (! check_word_is(fptr, "NUMBER_TRAINING_POINTS")) return NULL;
  m->nmodel_points = nmodel_points = read_integer(fptr);

  if (! check_word_is(fptr, "COVARIANCE_FUNCTION_INDEX")) return NULL;
  m->cov_fn_index = cov_fn_index = read_integer(fptr);

  if (! check_word_is(fptr, "REGRESSION_ORDER")) return NULL;
  m->regression_order = regression_order = read_integer(fptr);

  if (! check_word_is(fptr, "FIXED_NUGGET_MODE")) return NULL;
  m->fixed_nugget_mode = read_bool(fptr); /* only option. */

  if (! check_word_is(fptr, "FIXED_NUGGET")) return NULL;
  m->fixed_nugget = read_double(fptr); /* not used. */

  if (! check_word_is(fptr, "USE_DATA_SCALES")) return NULL;
  m->use_data_scales = read_bool(fptr);

  if (! check_word_is(fptr, "NUMBER_THETAS")) return NULL;
  m->number_thetas = read_integer(fptr);
  assert( m->number_thetas ==
  				((cov_fn_index == POWEREXPCOVFN) ? (nparams + 2) : 3));

  if (! check_word_is(fptr, "NUMBER_REGRESSION_FUNCTIONS")) return NULL;
  m->nregression_fns = read_integer(fptr);
  assert( m->nregression_fns == (1 + ((regression_order) * (nparams))));


  // now we can allocate everything in m
  m->xmodel = gsl_matrix_alloc(nmodel_points, nparams);
  m->training_matrix = gsl_matrix_alloc(nmodel_points, number_of_outputs);
  m->training_mean = gsl_vector_alloc(number_of_outputs);
  m->pca_evals_r = gsl_vector_alloc(number_of_pca_outputs);
  m->pca_evecs_r = gsl_matrix_alloc(number_of_outputs, number_of_pca_outputs);
  m->pca_zmatrix = gsl_matrix_alloc(nmodel_points, number_of_pca_outputs);
  m->sample_scales = gsl_vector_alloc(nparams);
  m->grad_ranges = gsl_matrix_alloc(m->number_thetas, 2);
  m->pca_model_array
    = (modelstruct**)malloc(sizeof(modelstruct*) * number_of_pca_outputs);
  ////////////////////////////

  if (! check_word_is(fptr, "PARAMETER_VALUES")) return NULL;
  gsl_matrix_fscanf(fptr, m->xmodel);

  if (! check_word_is(fptr, "OUTPUT_VALUES")) return NULL;
  gsl_matrix_fscanf(fptr, m->training_matrix);

  if (! check_word_is(fptr, "OUTPUT_PCA_EIGENVALUES")) return NULL;
  gsl_vector_fscanf(fptr, m->pca_evals_r);

  if (! check_word_is(fptr, "OUTPUT_PCA_EIGENVECTORS")) return NULL;
  gsl_matrix_fscanf(fptr, m->pca_evecs_r);

  if (! check_word_is(fptr, "PCA_DECOMPOSED_Z_VALUES")) return NULL;
  gsl_matrix_fscanf(fptr, m->pca_zmatrix);

  if (! check_word_is(fptr, "MEANS_OF_TRAINING_POINTS")) return NULL;
  gsl_vector_fscanf(fptr, m->training_mean);

  if (! check_word_is(fptr, "SAMPLE_SCALES")) return NULL;
  gsl_vector_fscanf(fptr, m->sample_scales);

  if (! check_word_is(fptr, "GRAD_RANGES")) return NULL;
  gsl_matrix_fscanf(fptr, m->grad_ranges);

  for(i = 0; i < number_of_pca_outputs; i++) {
    m->pca_model_array[i] = load_modelstruct_3(fptr,m,i);
    assert(m->pca_model_array[i] != NULL);
  }
  return m;
}

/**
 * return the sum of the elements of vec from 0:nstop
 */
double vector_elt_sum(gsl_vector* vec, int nstop)
{
  assert(nstop >= 0);
  assert((unsigned)nstop <= vec->size);
  int i;
  double sum = 0.0;
  for(i = 0; i < nstop; i++){
    sum += gsl_vector_get(vec, i);
  }
  return(sum);
}


/**
 * this free's everything in m
 */
void free_multimodelstruct(multi_modelstruct *m)
{
  int i;
  free_string_array(m->parameter_names);
  free_string_array(m->output_names);
  free(m->parameter_minima);
  free(m->parameter_maxima);
  gsl_vector_free(m->training_mean);
  for(i = 0; i < m->nr; i++){
    free_modelstruct_2(m->pca_model_array[i]);
    //gsl_matrix_free(m->pca_model_array[i]->xmodel);
  }
  free(m->pca_model_array);
  gsl_matrix_free(m->xmodel);
  gsl_matrix_free(m->training_matrix);
  gsl_vector_free(m->pca_evals_r);
  gsl_matrix_free(m->pca_evecs_r);
  gsl_matrix_free(m->pca_zmatrix);
}






