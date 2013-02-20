/**
multi_modelstruct.h

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

#ifndef __INC_MULTIMODELSTRUCT__
#define __INC_MULTIMODELSTRUCT__


#include "modelstruct.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

struct modelstruct;

/**
 * \struct multimodelstruct
 * handle a model with t-variate output i.e y_m = {y_1, ..., y_t}^{T}
 *
 * this contains some r <= t modelstructs representing the emulators
 * in the PCA projected space
 */
typedef struct multi_modelstruct{
  /**
   * number of observables in OBSERVED space
   */
  int nt;

  /**
   * Number of pca-observables kept in the ROTATED space
   */
  int nr;

  /**
   * number of dimensions in the PARAMETER SPACE.
   * The number of parameters in the model to be emulated, this sets
   * the dimensionality of the estimation space and the "hardness" of
   * the estimation process.
   */
  int nparams;

  /**
   * number of points in the design
   * how many points to evaluate the emulated model at
   */
  int nmodel_points;

  /**
   * array of c-strings of length nparams.
   * To make things easier, this array is NULL terminated.  It can be
   * free()ed with the free_string_array() function.
   */
  char ** parameter_names;

  /**
   * array of c-strings of length nt.
   * To make things easier, this array is NULL terminated.  It can be
   * free()ed with the free_string_array() function.
   */
  char ** output_names;

  /**
   * array of parameter minima of length nparams.  This information is
   * for external use only.  It is not used inside this program.
   */
  double * parameter_minima;

  /**
   * array of parameter maxima of length nparams.  This information is
   * for external use only.  It is not used inside this program.
   */
  double * parameter_maxima;

  int cov_fn_index;
  int regression_order;

  /**
   * the design (rows:nmodel_points) (cols:nparams)
   */
  gsl_matrix *xmodel;
  /**
   * the full training set in the original space
   * (rows:nmodel_points) (cols:nt)
   */
  gsl_matrix *training_matrix;
  /**
   * a t length vector of the the mean values of the cols of training_matrix
   */
  gsl_vector *training_mean;

  /**
   * array of (r) pca decomposed models
   *
   * pointers from this back to xmodel, is this bad?
   */
  modelstruct** pca_model_array;

  // /**
  //  * the eigenvalues and vectors from the pca decomp (t xt) , not
  //  * saved for now...
  //  */
  // gsl_vector *pca_eigenvalues;
  // gsl_matrix *pca_eigenvectors;

  /**
   * just the first r evals of the pca decomp
   */
  gsl_vector *pca_evals_r;

  /**
   * the first r evecs (t x nr) of the  pca decomp
   */
  gsl_matrix *pca_evecs_r;

  /**
   * an (r x nmodel_points) matrix of the pca-transformed training
   * points, these are used to init each of the r entries of
   * pca_model_array
   *
   * z <- (1/sqrt(pca_evals_r)) %*% t(pca_evecs_r)
   *              %*% (training_matrix - training_mean)
   */
  gsl_matrix *pca_zmatrix;

} multi_modelstruct;

multi_modelstruct* alloc_multimodelstruct(gsl_matrix *xmodel_in, gsl_matrix *training_matrix_in,
                                         int cov_fn_index, int regression_order, double varfrac);

void gen_pca_decomp(multi_modelstruct *m, double vfrac);
void gen_pca_model_array(multi_modelstruct *m);

void dump_multi_modelstruct(FILE* fptr, multi_modelstruct *m );
multi_modelstruct *load_multi_modelstruct(FILE* fptr);


double vector_elt_sum(gsl_vector* vec, int nstop);

void free_multimodelstruct(multi_modelstruct *m);

#endif
