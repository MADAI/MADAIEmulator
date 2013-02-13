#ifndef __INC_MULTIVARSUPPORT__
#define __INC_MULTIVARSUPPORT__

#include "multi_modelstruct.h"
#include "emulator_struct.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

struct multi_modelstruct;
struct emulator_struct;

typedef struct multi_emulator {
	/** number of original output variables **/
	int nt;
	/** number of principal component output variables  */
	int nr;
	/** number of input parameters  */
	int nparams;
	/** number of training points */
	int nmodel_points;
	/** not used */
	int nregression_fns;
	/** not used */
	int nthetas;
	
	multi_modelstruct *model;
	emulator_struct **emu_struct_array;
	
} multi_emulator;


multi_emulator *alloc_multi_emulator(multi_modelstruct *model);

void free_multi_emulator(multi_emulator *e);

void estimate_multi(multi_modelstruct *m, FILE* outfp);

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
void emulate_point_multi(multi_emulator *emu,
												 gsl_vector *the_point,
												 gsl_vector *the_mean, 
												 gsl_matrix *the_covariance);

void emulate_point_multi_pca(multi_emulator *emu, gsl_vector *the_point,
												 gsl_vector *the_mean, 
												 gsl_vector *the_variance);



#endif
