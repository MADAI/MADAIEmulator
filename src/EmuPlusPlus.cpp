#include "EmuPlusPlus.h"

#include <cmath>
#include <string>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdio> // for the c io to load the statefile

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

/* the c-interface to the emulator */
extern "C"{
#include "multivar_support.h"
#include "modelstruct.h"
#include "multi_modelstruct.h"
}

using namespace std;

/**
 * 22.06.2012, ccs
 * cec24@phy.duke.edu
 *
 * a c++ interface to an emulator, for use in MCMC or similar,
 * requries a trained emulator statefile as produced by interactive_emulator.c
 *
 * for reference to the interface see parameterset.cc and mcmc.cc in madai-stat
 *
 * interface should expose the following required fns:
 *
 * emulator(string StateFilePath, bool PcaOnly)
 * @argument filepath -> location of the emulator state file
 * @argument PcaOnly -> if true query emulator will return the results in the pca space (z)
 * without rotating them back to the true output space (y)
 *
 * creates an emulator object
 * filepath should point to the emulator state file
 *
 * DestroyEmulator(...)
 * kills everything nicely, the destructor
 *
 * QueryEmulator(ParameterSet xpoint, vector<double> &Means, vector<double> &Errors);
 * sample the emulator at the point in the parameter space xpoint,
 * @return: Means is a vector of the emulator means at xpoint
 * @return: Errors is a vector of the emulator error (sqrt(var)) at xpoint
 *
 * Optional (but useful) functions:
 *
 * GetEmulatorPCADecomp(...)
 * returns the evalues and evectors of the pca decomp for a multivariate emulator, need this to use the emulator
 * in pca-only mode.
 *
 */

// default constructor
emulator::emulator(const std::string & StateFilePath) :
	StateFilePath(StateFilePath)
{
	FILE *fptr = fopen(this->StateFilePath.c_str(), "r");
	if(fptr == NULL){
		cerr << "error opening efile: " << StateFilePath << endl;
		exit(EXIT_FAILURE); // could throw here?
	}
	// this will load the modelstruct
	this->the_model = load_multi_modelstruct(fptr);
	fclose(fptr);

	if(this->the_model == NULL){
		cerr << "error opening statefile: " << StateFilePath << endl;
		exit(EXIT_FAILURE); // could throw here?
	}

	this->the_emulator = alloc_multi_emulator(the_model);
	if(this->the_emulator == 0){
		cerr << "error initializing multi_emulator" << endl;
		exit(EXIT_FAILURE);
	}

	this->the_emulate_point = gsl_vector_alloc(this->the_model->nparams);
	int number_outputs = this->the_model->nt;
	this->the_emulate_mean = gsl_vector_alloc(number_outputs);
	this->the_emulate_covariance
		= gsl_matrix_alloc(number_outputs, number_outputs);

	int number_pca_outputs = this->the_model->nr;
	this->the_emulate_pca_mean = gsl_vector_alloc(number_pca_outputs);
	this->the_emulate_var = gsl_vector_alloc(number_pca_outputs);
}



/**
 * Query the emulator with a vector xpoint in the parameter space.
 * Returns the means and a (flattened) covaraince matrix.
 * The means will be in the original output space.
 *
 * Output is the Y values in the real space (length is getNumberOfOutputs())
 * Covariance length is getNumberOfOutputs()*getNumberOfOutputs().
 *
 * @argument xpoint: the location in the parameter space where to
 *                   sample the emulator
 * @return means: filled with the emulator mean for the outputs at the point
 * @return covariance: filled a flattened matrix of emulator
 *                     covariance for the outputs at a point
 */
void emulator::QueryEmulator(
	const std::vector<double> &xpoint,
	std::vector<double> &Means,
	std::vector<double> &Covariance) {

	int number_outputs = this->the_model->nt;
	int number_params = this->the_model->nparams;
	if(xpoint.size() != number_params){
		cerr
			<< "Error::QueryEmulator called with incorrect number"
			" of dimensions in xpoint" << endl
			<< "xpoint.length: " << xpoint.size()
			<< " emulator->number_params: " << number_params << endl;
		exit(EXIT_FAILURE);
	}

	// vector::resize() is a no-op if already correct size.
	Means.resize(number_outputs);
	// vector::resize() is a no-op if already correct size.
	Covariance.resize(number_outputs * number_outputs);

	// setup the gsl vectors
	for(int i = 0; i < number_params; i++){
		// copy in the xpoint cpts into the internal gsl vector
		gsl_vector_set(the_emulate_point, i, xpoint[i]);
	}

	gsl_vector_set_zero(the_emulate_mean); // is this necessary?
	gsl_matrix_set_zero(the_emulate_covariance);

	// call the emulator
	emulate_point_multi(the_emulator, the_emulate_point,
		the_emulate_mean, the_emulate_covariance);

	// now copy out the results
	for(int i = 0; i < number_outputs; i++){
		Means[i] = gsl_vector_get(the_emulate_mean, i);
	}
	int index = 0;
	for(int i = 0; i < number_outputs; i++) {
		for(int j = 0; j < number_outputs; j++) {
			Covariance[index] = gsl_matrix_get(the_emulate_covariance, i, j);
			++index; // flatten the matrix
		}
	}
}

/**
 * Query the emulator with a vector xpoint in the parameter space
 * Returns the means and a (flattened) covaraince matrix.
 * the means and variances will be in PCA space.
 *
 * query the emulator at a given point
 *
 * output is the Z values in PCA space (length is getNumberOfPCAOutputs())
 *
 * @argument xpoint: the location in the parameter space where to
 *                   sample the emulator
 * @return Means: filled with the emulator mean for the outputs at the
 *                point
 * @return Variance: filled with the emulator variance for the outputs
 *                   at a point
 */
void emulator::QueryEmulatorPCA(
	const std::vector<double> &xpoint,
	std::vector<double> &Means,
	std::vector<double> &Variance) {

	int number_pca_outputs = this->the_model->nr;
	int number_params = this->the_model->nparams;

	// this->the_emulate_pca_mean = gsl_vector_alloc(number_pca_outputs);
	// this->the_emulate_var = gsl_vector_alloc(number_pca_outputs);

	if(xpoint.size() != number_params){
		cerr
			<< "Error::QueryEmulator called with incorrect"
			" number of dimensions in xpoint" << endl
			<< "xpoint.length: " << xpoint.size()
			<< " emulator->number_params: " << number_params << endl;
		exit(EXIT_FAILURE);
	}

	// vector::resize() is a no-op if already correct size.
	Means.resize(number_pca_outputs);
	// vector::resize() is a no-op if already correct size.
	Variance.resize(number_pca_outputs);

	// setup the gsl vectors
	for(int i = 0; i < number_params; i++){
		// copy in the xpoint cpts into the internal gsl vector
		gsl_vector_set(the_emulate_point, i, xpoint[i]);
	}

	gsl_vector_set_zero(the_emulate_pca_mean);
	gsl_vector_set_zero(the_emulate_var);

	// do the output in the pca space
	emulate_point_multi_pca(the_emulator, the_emulate_point,
		the_emulate_pca_mean, the_emulate_var);

	// now copy out the results
	for(int i = 0; i < number_pca_outputs; i++){
		Means[i] = gsl_vector_get(the_emulate_mean, i);
		Variance[i] = gsl_vector_get(the_emulate_var, i);
	}
}

/**
 * get the pca decomp from the emulator, you'll need this to rotate your 'real' output into
 * the pca space if you want to compare
 *
 * the PCA decomp on a set Y of training data, each observation being a vector of length nt and
 * with the set Y being n long:
 * 1) compute the column mean:  pca_mean_j = (1/n) Sum_{i=1}^{n} Y_j
 * where pca_mean is then a vector of length nt.
 * 2) compute the observed covariance matrix (nt x nt)  pca_cov_matrix_ij = (1/n) (Y-pca_mean)_ik (Y-pca_mean)_kj
 * 3) eigendecompose this covarance matrix: pca_cov_matrix = U^{-1} LAMBDA U
 * where U is an (nt x nt) (column) matrix of the eigenvectors and LAMDBA is a diagonal (nt x nt) matrix of the
 * eigenvalues.
 *
 * @requires: the vectors have been allocated
 *
 * @return pca_evals is set to the eigenvalues of the pca-decomp, sorted in descending absolute value (is length nr)
 * @return pca_evecs is set to the eigenvectors of the pca-decomp, the eigenvectors are stored in the columns and sorted to match the eigenvalues (is (rows)nt by (columns)nr)
 * @return pca_mean is set to the mean of the training data (and is length nt)
 */

void emulator::getEmulatorPCA(vector<double> *pca_evals, vector< vector<double> > *pca_evecs,
															vector<double> *pca_mean){

	for(int i = 0; i < this->the_model->nr; i++){
		pca_evals->push_back(gsl_vector_get(this->the_model->pca_evals_r, i));
	}

	for(int i = 0; i < this->the_model->nt; i++){
		pca_mean->push_back(gsl_vector_get(this->the_model->pca_evals_r, i));
	}

	// pca_evecs must have been allocated...
	for(int i = 0; i< this->the_model->nt; i++){
		for(int j = 0; j < this->the_model->nr; j++){
			(*pca_evecs)[i][j] = gsl_matrix_get(this->the_model->pca_evecs_r, i, j);}
	}

}


/**
 * default destructor for the emulator
 */
emulator::~emulator(){
	// free the c-structures
	free_multi_emulator(the_emulator);
	//free_multimodelstruct(the_model); // this is a memory leaker  :(
	gsl_vector_free(the_emulate_point);
	gsl_vector_free(the_emulate_mean);
	gsl_vector_free(the_emulate_var);
	gsl_matrix_free(the_emulate_covariance);
}


