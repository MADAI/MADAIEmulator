#ifndef __INCLUDE_EMUPLUSPLUS__
#define __INCLUDE_EMUPLUSPLUS__



#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <string>
#include <vector>

extern "C"{
 #include "multi_modelstruct.h"
 #include "multivar_support.h"
}


/**
 * @file emuplusplus.h
 * \brief a simple c++ interface to a trained emulator
 *
 * Instance of the emulator class are built from a trained emulator,
 * the emulator can then be sampled at a point in the parameter space
 * by the function QueryEmulator
 *
 * They must be initialized from an interactive_emulator statefile,
 * this includes all the data for a trained emulator
 *
 */

class emulator{
 public:
	emulator(const std::string & StateFilePath);
	~emulator();

	/**
	 * Query the emulator with a vector xpoint in the parameter space
	 * Returns the means and a (flattened) covaraince matrix.
	 * the means and variances will be in PCA space.
	 */
	void QueryEmulatorPCA(
		const std::vector<double> &xpoint,
		std::vector<double> &Means,
		std::vector<double> &Variance);

	/**
	 * Query the emulator with a vector xpoint in the parameter space.
	 * Returns the means and a (flattened) covaraince matrix.
	 * the means will be in the original output space.
	 */
	void QueryEmulator(
		const std::vector<double> &xpoint,
		std::vector<double> &Means,
		std::vector<double> &Covariance);

	/**
	 * get the emulator pca decomp
	 */
	void getEmulatorPCA(
		std::vector<double> *pca_evals,
		std::vector< std::vector<double> > *pca_evecs,
		std::vector<double> *pca_mean);

	/**
	 * FIXME: description
	 */
	int getRegressionOrder() const { return this->the_model->regression_order; }
	/**
	 * FIXME: description
	 */
	int getCovFnIndex() const { return this->the_model->cov_fn_index; }

	/**
	 * FIXME: description
	 */
	int getNumberOfParameters() const { return this->the_model->nparams; }

	/**
	 * FIXME: description
	 */
	int getNumberOfOutputs() const { return this->the_model->nt; }

	/**
	 * FIXME: description
	 */
	int getNumberOfPCAOutputs() const { return this->the_model->nr; }

	/**
	 * FIXME: description
	 */
	const char * getParameterName(int index) const;

	/**
	 * FIXME: description
	 */
	double getParameterMinimum(int index) const;

	/**
	 * FIXME: description
	 */
	double getParameterMaximum(int index) const;

	/**
	 * FIXME: description
	 */
	const char * getOutputName(int index) const;

	/**
	 * FIXME: description
	 */
	bool IsOkay() const;

 private:
	std::string StateFilePath;
	multi_modelstruct *the_model;  // the c structure which defines the model
	multi_emulator *the_emulator; // the c structure which defines the emulator

	gsl_vector *the_emulate_point; // size == getNumberOfParameters()
	gsl_vector *the_emulate_mean; // size == getNumberOfOutputs()
	gsl_vector *the_emulate_pca_mean; // size == getNumberOfPCAOutputs()
	gsl_vector *the_emulate_var; // size == getNumberOfPCAOutputs()
	gsl_matrix *the_emulate_covariance;

};


#endif
