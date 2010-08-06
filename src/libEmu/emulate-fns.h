#ifndef _INC_EMULATE_FNS_
#define _INC_EMULATE_FNS_

#include "stdio.h"
#include "stdlib.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"

#include "emulator.h"
#include "regression.h"


#include "../optstruct.h"
#include "../modelstruct.h"
#include "../resultstruct.h"


void emulate_model_results(modelstruct *the_model, optstruct* options, resultstruct* results);
void emulate_ith_location(modelstruct *the_model, optstruct *options, resultstruct *results,int i, gsl_matrix* h_matrix, gsl_matrix* cinverse, gsl_vector *beta_vector);
void chol_inverse_cov_matrix(optstruct* options, gsl_matrix* temp_matrix, gsl_matrix* result_matrix, double* final_determinant_c);
#endif