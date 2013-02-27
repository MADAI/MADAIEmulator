/*********************************************************************
INTERACTIVE N-D GAUSSIAN PROCESS INTERPOLATOR (MODEL EMULATOR)
Copyright 2012, The University of North Carolina at Chapel Hill.

DESCRIPTION:
  Meant to be included in the GP-Emulator program
  <https://github.com/jackdawjackdaw/emulator> by C.Coleman-Smith
  <cec24@phy.duke.edu>, Copyright 2009-2011 Duke University.

ACKNOWLEDGMENTS:
  This software was written in 2012 by Hal Canary <cs.unc.edu/~hal>,
  based off of code written by Christopher Coleman-Smith
  <cec24@phy.duke.edu> in 2010-2012 while working for the MADAI project
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

USE:
  For details on how to use the interactive_emulator, consult the
  manpage via:

    $ nroff -man < [PATH_TO/]interactive_emulator.1 | less

  or, if the manual is installed:

    $ man 1 interactive_emulator

BUGS:
  Writing MODEL_SNAPSHOT_FILE to stdout is not allowed because
  GPEmulatorLib is excessively verbose.  This should be fixed when we
  refactor.

  This program probably should be two seperate executables, but for
  now it is self-contained in one source file.

  Multiple models can't be run simultaniously due to the use of global
  variables in GPEmulatorLib.  This is being fixed.

  "interactive_emulator estimate_thetas ..." is mostly redundant
  against "estimator", except that the input file format and
  command-line arguments are different.

BIGGEST BUG:
  What do I do when I have multiple training vectors (Y's) i.e. I'm
  trying to emulate several functions defined on the same space and
  sampled on the same training points?  We need to modify this to
  efficiently do that.

  this is somewhat supported now


 TODO:
  - allow the design x[0,0]...x[nmodel_points,nparams-1] to be
    read separately from the training points
  - check for scaling of design and training values, warn if they
    different columns have very different scales
  - make estimation process respect the quiet flag?

*********************************************************************/

/* #define BINARY_INTERACTIVE_MODE */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>

#include "main.h"
#include "useful.h"
#include "modelstruct.h"
#include "emulator_struct.h"
#include "multi_modelstruct.h"
#include "multivar_support.h"
#include "libEmu/maxmultimin.h"



/**
 * contents will be set by global_opt_parse, can be passed into
 * estimate_thetas and interactive_mode to be used appropriately
 */
struct cmdLineOpts{
  int regOrder;  /* -r --regression_order = */
  int covFn;     /* -c --covariance_fn = */
  int quietFlag; /* -q --quiet */
  int pcaOutputFlag; /* -z --pca_output  turns on pca only output */
  double pca_variance; /* -v --pca_variance =  fractional value for the pca decomp */

  // add additional flags as needed here

  // what mode to run in
  char* run_mode; /* arg 1 */
  char *inputfile; /* arg 2 (maybe) */
  // the state filename
  char* statefile; /* arg 2 or 3 */
} cmdLineOpts;


struct cmdLineOpts* global_opt_parse(int argc, char** argv);

/********************************************************************/
static const char useage [] =
  "useage:\n"
  "  interactive_emulator estimate_thetas INPUT_MODEL_FILE MODEL_SNAPSHOT_FILE [OPTIONS]\n"
  "or\n"
  "  interactive_emulator interactive_mode MODEL_SNAPSHOT_FILE [OPTIONS]\n"
  "or\n"
  "  interactive_emulator print_thetas MODEL_SNAPSHOT_FILE\n"
  "\n"
  "INPUT_MODEL_FILE can be \"-\" to read from standard input.\n"
  "\n"
  "The input MODEL_SNAPSHOT_FILE for interactive_mode must match the format of\n"
  "the output MODEL_SNAPSHOT_FILE from estimate_thetas.\n"
  "\n"
  "Options which only influence estimate_thetas:\n"
  "  --regression_order=0 (const)\n"
  "  --regression_order=1 (linear)\n"
  "  --regression_order=2 (quadratic)\n"
  "  --regression_order=3 (cubic)\n"
  "  --covariance_fn=0 (POWER_EXPONENTIAL)\n"
  "  --covariance_fn=1 (MATERN32)\n"
  "  --covariance_fn=2 (MATERN52)\n"
  "  (-v=FRAC) --pca_variance=FRAC : sets the pca decomp to keep cpts up to variance fraction frac\n"
  "options which incluence interactive_mode:\n"
  "  (-q) --quiet: run without any extraneous output\n"
  "  (-z) --pca_output: emulator output is left in the pca space\n"
  "general options:\n"
  "  -h -? print this dialogue\n"
  "The defaults are regression_order=0 and covariance_fn=POWER_EXPONENTIAL.\n";

/**
 * Convienence function for exiting a function.
 */
int perr(const char * s) {
  fprintf(stderr,"%s\n",s);
  return EXIT_FAILURE;
}

/**
 *
 * @param input_filename: name of file to open.
 *   can be "-" or "stdin" to represent standard input
 * @param cov_fn_index sety by the command-line argument
 * @param regression_order set by the command-line argument
 * @param pca_variance_fraction set by the command-line argument
 * @returns NULL on error.
 */
multi_modelstruct * load_training_file(
    char * input_filename,
    int cov_fn_index,
    int regression_order,
    double pca_variance_fraction) {
  FILE * fptr;
  int i, j, number_outputs, number_params, number_model_points;
  char ** parameter_names;
  double * parameter_minima;
  double * parameter_maxima;
  char ** output_names;
  gsl_matrix * xmodel;
  gsl_matrix * training_matrix;
  if ((0 == strcmp(input_filename, "-")) ||
      (0 == strcmp(input_filename, "stdin"))) {
    fptr = stdin;
  } else {
    fptr = fopen(input_filename, "r");
    if (fptr == NULL) { perror("fopen"); return NULL; }
  }
  discard_comments(fptr, '#');
  if (! check_word_is(fptr, "VERSION")) {
    fprintf(stderr, "wrong version string\n");
    return NULL;
  }
  int version_int = read_integer(fptr);
  if (version_int != 1) {
    fprintf(stderr, "wrong version!\n");
    return NULL;
  }

  /* PARAMETERS */
  if (! check_word_is(fptr, "PARAMETERS")) {
    fprintf(stderr, "syntax error: expected \"PARAMETERS\".\n");
    fclose(fptr);
    return NULL;
  }
  number_params = read_integer(fptr);
  assert(number_params > 0);
  parameter_names = allocate_string_array(number_params);
  parameter_minima = MallocChecked(number_params * sizeof(double));
  parameter_maxima = MallocChecked(number_params * sizeof(double));
  for (i = 0; i < number_params; ++i) {
    parameter_names[i] = read_word(fptr);
    parameter_minima[i] = read_double(fptr);
    parameter_maxima[i] = read_double(fptr);
  }

  /* OUTPUTS */
  if (! check_word_is(fptr, "OUTPUTS")) {
    fprintf(stderr, "syntax error: expected \"PARAMETERS\".");
    free_string_array(parameter_names);
    free(parameter_minima);
    free(parameter_maxima);
    fclose(fptr);
    return NULL;
  }
  number_outputs = read_integer(fptr);
  assert (number_outputs > 0);
  output_names = allocate_string_array(number_outputs);
  for (i = 0; i < number_outputs; ++i) {
    output_names[i] = read_word(fptr);
  }
  number_model_points = read_integer(fptr);
  assert(number_model_points > 0);

  xmodel = gsl_matrix_alloc(number_model_points, number_params);
  training_matrix = gsl_matrix_alloc(number_model_points, number_outputs);
  for(i = 0; i < number_model_points; ++i)
    for (j = 0; j < number_params; ++j)
      gsl_matrix_set(xmodel, i, j, read_double(fptr));
  for(i = 0; i < number_model_points; ++i)
    for (j = 0; j < number_outputs; ++j)
      gsl_matrix_set(training_matrix, i, j, read_double(fptr));
  fclose(fptr);

  pca_variance_fraction = clamp(pca_variance_fraction,0.0,1.0);
  cov_fn_index = clamp(cov_fn_index, 0, 3);
  regression_order = clamp(regression_order, 0, 3);

  /* Allocate the multi-model. Do the PCA decomp.
   * This is a little chatty on stderr */
  multi_modelstruct * model = alloc_multimodelstruct(
    xmodel, training_matrix,
    cov_fn_index, regression_order, pca_variance_fraction);
  if(model == NULL) {
    fprintf(stderr, "error in alloc_multimodelstruct()\n");
    free_string_array(parameter_names);
    free_string_array(output_names);
    free(parameter_minima);
    free(parameter_maxima);
    gsl_matrix_free(xmodel);
    gsl_matrix_free(training_matrix);
    return NULL;
  }
  model->parameter_names = parameter_names;
  model->parameter_minima = parameter_minima;
  model->parameter_maxima = parameter_maxima;
  model->output_names = output_names;
  return model;
}



/**
 * If main called with "estimate_thetas_multi"
 * assume all cmdOpts correctly filled.
 */
int estimate_thetas(struct cmdLineOpts* cmdOpts) {
  assert(cmdOpts != NULL);
  multi_modelstruct * model = load_training_file(
    cmdOpts->inputfile,
    cmdOpts->covFn,
    cmdOpts->regOrder,
    cmdOpts->pca_variance);
  if (model == NULL)
    return perr("Input file read failed.\n");

  FILE * outfp = fopen(cmdOpts->statefile, "w");
  if (outfp == NULL)
    return perr("Opening statefile failed.");

  /** this will write the model state file to outfp */
  estimate_multi(model, outfp);
  fclose(outfp);
  free_multimodelstruct(model);
  return EXIT_SUCCESS;
}


/**
 * If main called with "interactive_mode" argument, this happens.
 */
int interactive_mode (struct cmdLineOpts* cmdOpts) {
  FILE * interactive_input = stdin;
  FILE * interactive_output = stdout;
  int i, j, r, expected_r;

  FILE * fp = fopen(cmdOpts->statefile,"r");
  if (fp == NULL)
    return perr("Error opening file");
  multi_modelstruct *model = load_multi_modelstruct(fp);

  fclose(fp);

  multi_emulator *the_multi_emulator = alloc_multi_emulator(model);

  int number_params = model->nparams;
  gsl_vector * the_point = gsl_vector_alloc(number_params);

  int number_outputs = model->nt;
  int number_outputs_pca = model->nr;

  gsl_vector *the_mean = gsl_vector_alloc(number_outputs);
  gsl_matrix *the_covariance = NULL;
  gsl_vector *the_variance = NULL;

  int use_pca = ( cmdOpts->pcaOutputFlag != 0); /* C-style boolean */

  if(use_pca) {
    /* principal component output basis */
    the_variance = gsl_vector_alloc(number_outputs_pca);
    assert (the_variance != NULL);
  } else {
    /* original output basis */
    the_covariance = gsl_matrix_alloc(number_outputs, number_outputs);
    assert (the_covariance != NULL);
  }

  #ifdef BINARY_INTERACTIVE_MODE
  r = expected_r = sizeof(double);
  #else
  r = expected_r = 1;
  #endif

  if(!cmdOpts->quietFlag){
    fprintf(interactive_output, "VERSION 1\n");
    fprintf(interactive_output, "PARAMETERS\n");
    fprintf(interactive_output,"%d\n",number_params);
    for(i = 0; i < number_params; i++) {
      fprintf(interactive_output,"%s\n",model->parameter_names[i]);
      fprintf(interactive_output,"%.17f\n",model->parameter_minima[i]);
      fprintf(interactive_output,"%.17f\n",model->parameter_maxima[i]);
    }
    fprintf(interactive_output, "OUTPUTS\n");

    if(use_pca) {
      /* PCA output basis */
      fprintf(interactive_output,"%d\n",number_outputs_pca);
      for(i = 0; i < number_outputs_pca; i++) {
        fprintf(interactive_output,"Principal_Component_%d\n",i);
      }
      fprintf(interactive_output, "VARIANCE\n");
      fprintf(interactive_output, "%d\n", number_outputs_pca);
    } else {
      /* original output basis */
      fprintf(interactive_output,"%d\n",number_outputs);
      for(i = 0; i < number_outputs; i++) {
        fprintf(interactive_output,"%s\n",model->output_names[i]);
      }
      fprintf(interactive_output, "COVARIANCE\n");
      fprintf(interactive_output, "TRIANGULAR_MATRIX\n");
      /*
        For example, a 5x5 symmetric matrix can be represented with
        (5*(5+1)/2)=15 numbers.  If the layout of the matrix is:
            1 2 3 4 5
            2 6 7 8 9
            3 7 a b c
            4 8 b d e
            5 9 c e f
        Then we will serialize it as:
            1 2 3 4 5 6 7 8 9 0 a b c d e f
        To save time.
			*/
      fprintf(interactive_output, "%d\n",
        (number_outputs * (number_outputs + 1)) / 2 );
    }
    fprintf(interactive_output, "END_OF_HEADER\n");
    fflush(interactive_output);
  }

  while (! feof(interactive_input)) {
    for(i =0; (i < number_params) && (r == expected_r); i++) {
      #ifdef BINARY_INTERACTIVE_MODE
        r = fread(gsl_vector_ptr(the_point, i), sizeof(double), 1,
          interactive_input);
      #else
        r = fscanf(interactive_input, "%lf%*c", gsl_vector_ptr(the_point, i));
      #endif
    }
    if (r < expected_r) /* probably eof, otherwise error */
      break;
    if(use_pca) {
      /* support output in the pca space */
      assert (the_variance != NULL);
      emulate_point_multi_pca(
        the_multi_emulator, the_point, the_mean, the_variance);

      #ifdef BINARY_INTERACTIVE_MODE
        for(i = 0; i < number_outputs_pca; i++) {
          fwrite(gsl_vector_ptr(the_mean,i), sizeof(double), 1,
            interactive_output);
        }
        for(i = 0; i < number_outputs_pca; i++) {
          fwrite(gsl_vector_ptr(the_variance,i), sizeof(double), 1,
            interactive_output);
        }
      #else
        for(i = 0; i < number_outputs_pca; i++) {
          fprintf(interactive_output, "%.17f\n", gsl_vector_get(the_mean,i));
        }
        for(i = 0; i < number_outputs_pca; i++) {
          fprintf(interactive_output, "%.17f\n",
            gsl_vector_get(the_variance,i));
        }
      #endif
    } else {
      /* NOT use_pca */
      assert (the_covariance != NULL);
      emulate_point_multi(
        the_multi_emulator, the_point, the_mean, the_covariance);
      #ifdef BINARY_INTERACTIVE_MODE
        for(i = 0; i < number_outputs; i++) {
          fwrite(gsl_vector_ptr(the_mean,i), sizeof(double), 1,
            interactive_output);
        }
        for(i = 0; i < number_outputs; i++) {
          for(j = i; j < number_outputs; j++) {
            fwrite(gsl_matrix_ptr(the_covariance,i,j), sizeof(double), 1,
              interactive_output);
          }
        }
      #else
        for(i = 0; i < number_outputs; i++) {
          fprintf(interactive_output, "%.17f\n", gsl_vector_get(the_mean,i));
        }
        for(i = 0; i < number_outputs; i++) {
          for(j = i; j < number_outputs; j++) {
            fprintf(interactive_output, "%.17f\n",
              gsl_matrix_get(the_covariance,i,j));
          }
        }
      #endif
    }
    fflush(interactive_output);
  }

  free_multi_emulator(the_multi_emulator);
  /* this is causing segfaults, i guess i dont understand the
     allocation pattern here properly.
     TODO: run valgrind. */
  /* free_multimodelstruct(model); */
  gsl_vector_free(the_point);
  gsl_vector_free(the_mean);

  if(use_pca) {
    assert (the_variance != NULL);
    gsl_vector_free(the_variance);
  } else {
    assert (the_covariance != NULL);
    gsl_matrix_free(the_covariance);
  }
  return 0;
}

/**
 * If main called with "print_thetas" argument this happens.
 */
int print_thetas(struct cmdLineOpts *cmdOpts)
{
  int i,j, k;
  int nthetas; // number of thetas
  int nr; // number of emulators
  int nparams;
  int nmodel_points;
  double vartot = 0;
  char buffer[256];

  FILE * fp = fopen(cmdOpts->statefile,"r");
  if (fp == NULL)
    return perr("Error opening file");
  multi_modelstruct *model = load_multi_modelstruct(fp); // load the statefile

  fclose(fp);

  nr = model->nr;
  nparams = model->nparams;
  nthetas = model->pca_model_array[0]->thetas->size;
  nmodel_points = model->nmodel_points;


  printf("#-- EMULATOR LENGTH SCALES (thetas) IN PCA SPACE -- #\n");
  for(i = 0; i < nr ; i ++)
    vartot += gsl_vector_get(model->pca_evals_r,i);

  printf("#-- id\tpca-var\tScale\tNugget");
  for(i = 0; i < nparams; i++)
    printf("\tlength_%d", i);
  printf(" -- #\n");

  for(i = 0; i < nr; i++){
    printf("%d\t", i);
    printf("%lf\t", gsl_vector_get(model->pca_evals_r,i)/vartot);
    for(j = 0; j < nthetas; j++){
      printf("%lf\t", exp(gsl_vector_get(model->pca_model_array[i]->thetas, j)));
    }
    printf("\n");
  }

  // this produces full debug output for each of the nr emulators
  for(i = 0; i < nr; i++){
    sprintf(buffer, "pca_emu_summary_%d.dat", i);
    fp = fopen(buffer, "w");
    for(j = 0; j < nmodel_points; j++){
      for(k = 0; k < nparams; k++)
        fprintf(fp, "%lf\t", gsl_matrix_get(model->pca_model_array[i]->xmodel, j, k));
      fprintf(fp, "%lf\n", gsl_vector_get(model->pca_model_array[i]->training_vector, j));
    }
    fclose(fp);
  }



  return 0;
}






/*********************************************************************
main
*********************************************************************/
int main (int argc, char ** argv) {

  if(argc < 3) { // the minimum possible number of args is 3?
    return perr(useage);
  }

  struct cmdLineOpts *opts = global_opt_parse(argc, argv);

  if (str_equal(opts->run_mode, "estimate_thetas")){
    estimate_thetas(opts);
  } else if (str_equal(opts->run_mode, "interactive_mode")) {
    interactive_mode(opts);
  } else if (str_equal(opts->run_mode, "print_thetas")){
    print_thetas(opts);
  } else{
    free(opts);
    return perr(useage);
  }
  free(opts);
  return(EXIT_SUCCESS);
}


/**
 * option parsing using getoptlong
 */
struct cmdLineOpts* global_opt_parse(int argc, char** argv)
{

  /* note: flags followed with a colon come with an argument */
  static const char *optString = "r:c:qh?";

  // should add a variance option for the pca_decomp
  // and a flag to do return output in pca space
  static const struct option longOpts[] = {
    { "regression_order", required_argument , NULL, 'r'},
    { "covariance_fn", required_argument , NULL, 'c'},
    { "pca_variance", required_argument, NULL , 'v'}, // set the var fraction for the pca decomp
    { "pca_output",  no_argument , NULL , 'z'},  // output from interactive emulator is left in pca space
    { "quiet", no_argument , NULL, 'q'},
    { "help", no_argument , NULL, 'h'},
    { NULL, no_argument, NULL, 0}
  };

  struct cmdLineOpts *opts = (struct cmdLineOpts*) malloc(sizeof(struct cmdLineOpts));
  // init with default values
  opts->regOrder = 0;
  opts->covFn = 0;
  opts->quietFlag = 0;
  opts->pca_variance = 0.99;
  opts->pcaOutputFlag = 0; // if this is one, output multivar results without rotation
  opts->run_mode = NULL;
  opts->inputfile = NULL;
  opts->statefile = NULL;

  int longIndex;
  int opt;

  // add more options here
  opt = getopt_long( argc, argv, optString, longOpts, &longIndex );
  while( opt != -1 ) {
    switch( opt ) {
    case 'r':
      opts->regOrder = atoi(optarg);
      break;
    case 'c':
      opts->covFn = atoi(optarg); /* this expects the cov fn to be 0,1,2? */
      break;
    case 'v':
      opts->pca_variance = atof(optarg); /* expect the var to be a float < 1 > 0 */
      if(opts->pca_variance < 0.0 || opts->pca_variance > 1.0){
        fprintf(stderr, "# err pca_variance argument given incorrect value: %lf\n", opts->pca_variance);
        opts->pca_variance = 0.95;
        fprintf(stderr, "# using default value: %lf\n", opts->pca_variance);
      }
      fprintf(stderr, "# var-frac: %lf\n", opts->pca_variance);
    case 'z':
      opts->pcaOutputFlag = 1;
    case 'q':
      opts->quietFlag = 1;
      break;
    case 'h':   /* fall-through is intentional */
    case '?':
      //display_usage();
      exit(perr(useage));
      break;
    default:
      /* You won't actually get here. */
      break;
    }
    opt = getopt_long( argc, argv, optString, longOpts, &longIndex );
  }

  // set the remaining fields
  opts->run_mode = argv[optind];
  if(str_equal(opts->run_mode, "estimate_thetas")){
    opts->inputfile = argv[optind+1];
    opts->statefile = argv[optind+2];
  } else {
    opts->statefile = argv[optind+1];
  }

  /* If options are out of range, fix them, but let the user know! */
  if (opts->pca_variance > 1.0) {
    fprintf(stderr, "# ERROR: pca_variance too high.  Using 0.99.\n");
    opts->pca_variance = 0.99;
  }
  if (opts->pca_variance < 0.0) {
    fprintf(stderr, "# ERROR: pca_variance too low.  Using 0.99.\n");
    opts->pca_variance = 0.99;
  }
  if (opts->covFn < 0 || opts->covFn > 3){
    fprintf(stderr, "# ERROR: cov_fn_index %d not supported\n", opts->covFn);
    fprintf(stderr, "# Using default of 0.\n");
    opts->covFn = 0;
  }
  if (opts->regOrder < 0 || opts->regOrder > 3){
    fprintf(stderr, "# ERROR: regression_order %d not supported\n"
            "# Using default of 0.",
            opts->regOrder);
    opts->regOrder = 0;
  }
  return opts;
}
