/**
useful.h

Copyright 2009-2013, Duke University and The University of North
Carolina at Chapel Hill.

ACKNOWLEDGMENTS:
  This software was written in 2009-2013 by Hal Canary
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

/**
 * \date 11-1-09
 * \file useful.h,
 * \brief some basic functions one always uses in scientific code
 */

#include "gsl/gsl_rng.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"

void print_vector_quiet(gsl_vector *x, int n);
//******************
// tries to get a seed from /dev/random or otherwise uses the system time
//*****************
unsigned long int get_seed(void);
unsigned long int get_seed_noblock(void);

// errors
void *MallocChecked(size_t size);

/**
   Makes use of fscanf(fptr, "%d%*c",...) to read a single integer.

   If assert statements are enabled (! NDEBUG), it will give an
   error if anything goes wrong.
 */
int read_integer(FILE *fptr);

/**
   Makes use of fscanf(fptr, "%lf%*c",...) to read a single double.
   If assert statements are enabled (! NDEBUG), it will give an error
   if anything goes wrong.
 */
double read_double(FILE *fptr);

/**
 * Return true only if the beginning of s1 matches s2.
 */
#define starts_with(s1,s2) (strncmp ((s1), (s2), strlen(s2)) == 0)

/**
 * Return true only if s1 equals s2.
 */
#define str_equal(s1,s2) (strcmp ((s1), (s2)) == 0)

/**
 * returns x if (min <= x && x <= max)
 *   else returns min or max.
 */
#define clamp(x,min,max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))


/**
 * Returns TRUE if the next word in the file stream is equal to str;
 * else returns FALSE.
 */
int check_word_is(FILE *fptr, const char * str);

/**
 * Makes use of fscanf(fptr, "%s",...) to read a single
 * (whitespace-delimited) word.  Maximum length is 4095 characters.
 * Stores the word in newly-allocated space on the stack.  Be sure to
 * free() the string when you are done with it!
 */
char * read_word(FILE *fptr);

/**
 * Read and discard characters from stream fptr until a newline or the
 * end-of-file is reached.
 */
void discard_line( FILE * fptr );

/**
 * While the first letter of each line is a comment character, discard
 * lines.
 */
void discard_comments( FILE * fptr, char comment_character );

/**
 * Allocate a NULL-terminated array of char* pointers.
 */
char ** allocate_string_array(int size);

/**
 * Free a NULL-terminated array of char* pointers.
 * If the argument is NULL, nothing will happen.
 */
void free_string_array(char ** string_array);

#define print_int(fptr, d) fprintf((fptr), "%d\n", (d));
#define print_str(fptr, s) fprintf((fptr), "%s\n", (s));
#define print_double(fptr, x) fprintf((fptr), "%.17f\n", (x));
#define print_doubles(fptr, x) fprintf((fptr), "%.17f ", (x));

/**
 * returns "true" or "false"
 */
const char * int_to_bool(int i);
/**
 * return true iff next word is not 0 or "false" or "FALSE" or "False"
 */
int read_bool(FILE *fptr);

/**
 * FIXME doc
 */
void gsl_matrix_fprint_block(FILE * fptr, gsl_matrix * m);
