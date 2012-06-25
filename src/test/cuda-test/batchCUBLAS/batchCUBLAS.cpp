/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* This example demonstrates how to get better performance by
 * CUBLAS calls by using streams
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#if defined(_WIN32) 
#include <float.h>
#endif

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cublas_v2.h>
#include <shrQATest.h>

#include "batchCUBLAS.h"

const char *sSDKname = "batchCUBLAS";

//============================================================================================
// Device information utilities
//============================================================================================

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

int getDeviceVersion (void) {
    int device;
    struct cudaDeviceProp properties;
    if (cudaGetDevice (&device) != cudaSuccess) {
        printf ("failed to get device\n");
        return 0;
    }
    if (cudaGetDeviceProperties (&properties, device) != cudaSuccess) {
        printf ("failed to get properties\n");
        return 0;
    }        
    return properties.major * 100 + properties.minor * 10;
}

size_t getDeviceMemory (void) {
    struct cudaDeviceProp properties;
    int device;
    if (cudaGetDevice (&device) != cudaSuccess) {
        return 0;
    }
    if (cudaGetDeviceProperties (&properties, device) != cudaSuccess) {
        return 0;
    }        
    return properties.totalGlobalMem;
}
#if defined(__cplusplus)
}
#endif /* __cplusplus */

//============================================================================================
// random utilities
//============================================================================================

template < typename T_ELEM>
void  fillupMatrix( T_ELEM *A , int lda , int rows, int cols, int seed = 0 );

template <typename T_ELEM>  
void fillupMatrix( T_ELEM *A , int lda , int rows, int cols, int seed )
{   
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {                           
            A[i + lda*j ] = cuGet<T_ELEM> ( ((double)(((lda*i+j+seed) % 253)+1))/256.0, ((double)((((cols*i+j) + 123 + seed) % 253)+1))/256.0 );            
        }
    }
}
/* Explicit instantiation */
template void  fillupMatrix<float>( float *A , int lda , int rows, int cols, int seed );
template void  fillupMatrix<double>( double *A , int lda , int rows, int cols, int seed );

/* For debugging */
void printCuType( const char *str, float A )
{
   fprintf(stdout, "%s (0x%08x, %g)", str, floatAsUInt(A), A);
}

void printCuType( const char *str, double A )
{
   fprintf(stdout, "%s (0x%016llx, %g)", str, doubleAsULL(A), A);
}

//============================================================================================
// defines and structures
//============================================================================================

#define CUBLAS_SGEMM_MAX_ULP_ERR    (.3)
#define CUBLAS_DGEMM_MAX_ULP_ERR    (1.e-3)
#define CUBLAS_SGEMM_MAX_RELATIVE_ERR    (6.e-6)
#define CUBLAS_DGEMM_MAX_RELATIVE_ERR    (0.0)
#define CUBLAS_GEMM_TEST_COUNT     (30)
#define BENCH_MATRIX_M              (128)
#define BENCH_MATRIX_K              (128)
#define BENCH_MATRIX_N              (128)

#define CLEANUP()                       \
do {                                    \
    if (A) free (A);                    \
    if (B) free (B);                    \
    if (C) free (C);                    \
    fflush (stdout);                    \
} while (0)

struct  gemmOpts {
    int m;
    int n;
    int k;
    int useStream;
    char *elem_type;   
    int N;    // number of multiplications
};

template<typename T_ELEM> 
struct gemmTestParams {
    cublasOperation_t transa; 
    cublasOperation_t transb;
    int   m;
    int   n;
    int   k;
    T_ELEM alpha;
    T_ELEM beta;
};

//============================================================================================
// template wrappers for cuda functions
//============================================================================================

static inline cublasStatus_t cublasXgemm (cublasHandle_t handle,
                     cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, 
                     float *alpha, const float *A, int lda, 
                     float *B, int ldb, float *beta, 
                     float *C, int ldc)
{
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}                     
                     
static inline cublasStatus_t cublasXgemm (cublasHandle_t handle,
                     cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, 
                     double *alpha, const double *A, int lda, 
                     double *B, int ldb, double *beta, 
                     double *C, int ldc)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
} 

//============================================================================================
// Primary Application code
//============================================================================================

static int processArgs (int argc, char *argv[], struct gemmOpts *opts)
{
    int error = 0;
    int oldError;
    memset (opts, 0, sizeof(*opts));
    static char default_type[] = "d"; //default double
    opts->elem_type = default_type; 
    opts->N = 10;
    while (argc) {
        oldError = error;
        if (*argv[0] == SWITCH_CHAR) {
            switch (*(argv[0]+1)) {
            case 'm':
                opts->m = (int)atol(argv[0]+2);
                break;
            case 'n':
                opts->n = (int)atol(argv[0]+2);
                break;
            case 'k':
                opts->k = (int)atol(argv[0]+2);
                break;
            case 'N':
                opts->N = (int)atol(argv[0]+2);
                break;
            default:
                break;
            }
        }
        if (error > oldError) {
            fprintf (stderr, "Invalid switch '%c%s'\n",SWITCH_CHAR, argv[0]+1);
        }
        argc -= 1;
        argv++;
    }
    return error;
}

template <typename T_ELEM> 
static int TESTGEN(gemm) (const struct gemmOpts *opts,
                           int matrixM, int matrixN, int matrixK, int &numTests,
                           struct gemmTestParams<T_ELEM> *params)
{
    static T_ELEM alpha[] = { cuGet<T_ELEM>(0,0), cuGet<T_ELEM>(-1,-1), cuGet<T_ELEM>(1,-2), cuGet<T_ELEM>(2,-1), cuGet<T_ELEM>(0,-3) };
    static T_ELEM beta[]  = { cuGet<T_ELEM>(0,0), cuGet<T_ELEM>(-1,-1), cuGet<T_ELEM>(1,-2), cuGet<T_ELEM>(2,-1), cuGet<T_ELEM>(0,-3)};
    
#define NBR_ALPHAS (sizeof(alpha) / sizeof(alpha[0]))
#define NBR_BETAS (sizeof(beta) / sizeof(beta[0]))
    static T_ELEM theAlpha;
    static T_ELEM theBeta;
    static int state;
    static int m;
    static int n;
    static int k;

    if (numTests-- <= 0)return -1;

    theAlpha = alpha[cuRand()%NBR_ALPHAS];
    theBeta  = beta[cuRand()%NBR_BETAS];        
    params->transa = CUBLAS_OP_N;
    params->transb = CUBLAS_OP_N;
    m = matrixM;
    n = matrixN;
    k = matrixK;
    params->m = m;
    params->n = n;
    params->k = k;
    params->alpha = theAlpha; 
    params->beta = theBeta;
    
    printf ("#### args: ta=%c tb=%c m=%d n=%d k=%d ",
            params->transa, params->transb, params->m, params->n, 
            params->k);
    printCuType( " alpha =", params->alpha);
    printCuType( " beta=",params->beta);
    printf("\n");                

    m = cuRand() % matrixM;
    n = cuRand() % matrixN;
    k = cuRand() % matrixK;

    state = cuRand() % 9;
    return 0;
}

template <typename T_ELEM>  
void fillupMatrixDebug( T_ELEM *A , int lda , int rows, int cols )
{   
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {                           
            A[i + lda*j ] = cuGet<T_ELEM> ( i + j );            
        }
    }
}

template <typename T_ELEM>  
int test_gemm_loop( struct gemmOpts &opts, float err, double max_relative_error, cublasHandle_t handle )
{
    struct gemmTestParams<T_ELEM> params;
    cudaStream_t *streamArray = 0;        
    cublasStatus_t status1, status2, status3;
    T_ELEM *A = NULL;
    T_ELEM *B = NULL;
    T_ELEM *C = NULL;
    T_ELEM **devPtrA = 0;
    T_ELEM **devPtrB = 0;
    T_ELEM **devPtrC = 0;
    int matrixM, matrixN, matrixK;
    int rowsA, rowsB, rowsC;
    int colsA, colsB, colsC;
    int matrixSizeA, matrixSizeB, matrixSizeC;
    int errors;
    double start, stop;
//  char *devPtrReserve = 0; 
    
    printf( "Testing %cgemm\n", *opts.elem_type);

    matrixM = (opts.m) ? opts.m : BENCH_MATRIX_M;
    matrixN = (opts.n) ? opts.n : BENCH_MATRIX_N;
    matrixK = (opts.k) ? opts.k : BENCH_MATRIX_K;
    
    rowsA = imax (1, matrixM);
    colsA = imax (1, matrixK);
    rowsB = imax (1, matrixK);
    colsB = imax (1, matrixN);
    rowsC = imax (1, matrixM);
    colsC = imax (1, matrixN);
    
    matrixSizeA = rowsA * colsA;
    matrixSizeB = rowsB * colsB;
    matrixSizeC = rowsC * colsC;

    devPtrA =(T_ELEM **)malloc (opts.N * sizeof(*devPtrA));
    devPtrB =(T_ELEM **)malloc (opts.N * sizeof(*devPtrB));
    devPtrC =(T_ELEM **)malloc (opts.N * sizeof(*devPtrC));
    
    for ( int i = 0; i < opts.N ; i++) {
        cudaError_t err1 = cudaMalloc ((void**)&devPtrA[i], matrixSizeA * sizeof(devPtrA[0][0]));
        cudaError_t err2 = cudaMalloc ((void**)&devPtrB[i], matrixSizeB * sizeof(devPtrB[0][0]));
        cudaError_t err3 = cudaMalloc ((void**)&devPtrC[i], matrixSizeC * sizeof(devPtrC[0][0]));
        if ((err1 != cudaSuccess) ||
            (err2 != cudaSuccess) ||
            (err3 != cudaSuccess)) { 
            CLEANUP();
            fprintf (stderr, "!!!! GPU memory allocation error\n");
            return CUBLASTEST_FAILED;                
        }                           
    }
    
    A  = (T_ELEM *)malloc (matrixSizeA * sizeof(A[0]));
    B  = (T_ELEM *)malloc (matrixSizeB * sizeof(B[0]));
    C  = (T_ELEM *)malloc (matrixSizeC * sizeof(C[0]));

    if ((!A) || (!B) || (!C)) {
        CLEANUP();
        fprintf (stderr, "!!!! system memory allocation error\n");
        return CUBLASTEST_FAILED;
    }
    
    streamArray = (cudaStream_t *)malloc(opts.N * sizeof (cudaStream_t *));
    
    for ( int i = 0; i < opts.N ; i++) {
        if (opts.useStream) {
            cudaError_t cudaErr = cudaStreamCreate(&streamArray[i]);    
            if (cudaErr != cudaSuccess){
                CLEANUP();
                fprintf (stderr, "!!!! cannot create stream\n");
                return CUBLASTEST_FAILED;
            }      
        }
        else {
            streamArray[i] = 0;
        }
    }       
                
    errors = 0;
    int numTests = 1;
    while (TESTGEN(gemm)(&opts, matrixM, matrixN, matrixK, numTests, &params) == 0) {
        printf( "#### args: lda=%d ldb=%d ldc=%d\n", rowsA, rowsB, rowsC);
        
        // fillup with Nan first (so lda padding is full on Nan)
        memset( A, 0xFF, matrixSizeA* sizeof(A[0]));     
        fillupMatrixDebug(A, rowsA, params.m, params.k);
        memset( B, 0xFF, matrixSizeB* sizeof(B[0]));        
        fillupMatrix(B, rowsB, params.k, params.n, 121);
      
        if (!cuEqual(params.beta, cuGet<T_ELEM>(0))) {
            fillupMatrix(C, rowsC, params.m, params.n);           
        } else {
              /* fill with SNaNs to make sure ZGEMM doesn't access C */
              memset(C, 0xFF, matrixSizeC * sizeof(C[0]) );   
        }
        double flopsCoef = 2.0;

        for ( int i = 0; i < opts.N ; i++) {
            status1 = cublasSetMatrix (rowsA, colsA, sizeof(A[0]), A, rowsA, devPtrA[i], rowsA);
            status2 = cublasSetMatrix (rowsB, colsB, sizeof(B[0]), B, rowsB, devPtrB[i], rowsB);
            status3 = cublasSetMatrix (rowsC, colsC, sizeof(C[0]), C, rowsC, devPtrC[i], rowsC);
            if ((status1 != CUBLAS_STATUS_SUCCESS) || (status2 != status1) || (status3 != status1)) {
                CLEANUP();
                fprintf (stderr, "!!!! GPU access error (write)\n");
                return CUBLASTEST_FAILED;
            }
        }

        start = second();
        start = second();
        for ( int i = 0; i < opts.N ; i++) {
            cublasSetStream(handle, streamArray[i]);
            status1 = cublasXgemm (handle, params.transa, params.transb, params.m, params.n, 
                                   params.k, &params.alpha, devPtrA[i], rowsA, 
                                   devPtrB[i], rowsB, &params.beta, devPtrC[i], rowsC);
            if (status1 != CUBLAS_STATUS_SUCCESS) {
                cudaError_t cudaStatus = cudaGetLastError();
                CLEANUP();
                fprintf (stderr, "!!!! GPU program execution error : cublas Error=%d, cuda Error=%d,(%s)\n", status1, cudaStatus,cudaGetErrorString(cudaStatus));
                return CUBLASTEST_FAILED;
            }
        }                     

        cudaError_t cudaStatus = cudaThreadSynchronize();
        if (cudaStatus != cudaSuccess) {
            CLEANUP();
            fprintf( stderr, "!!!! GPU program execution error on cudaThreadSynchronize : cudaError=%d,(%s)\n", cudaStatus,cudaGetErrorString(cudaStatus));
            return CUBLASTEST_FAILED;
        }

        stop = second();
       
        fprintf (stdout, "^^^^ elapsed = %10.8f sec  GFLOPS=%g\n", (stop-start), 
                 opts.N * (1e-9*flopsCoef*params.m*params.n*params.k)/(stop-start));

    } // end while (TESTGEN..

    CLEANUP();
    fprintf (stdout, "@@@@ %cgemm test %s\n", *opts.elem_type ,errors ? "FAIL" : "OK");
    return CUBLASTEST_PASSED;
}


int main (int argc, char *argv[])
{
    struct gemmOpts opts;
    int errors, nTimes, nTotalErrors = 0;
    int status = CUBLASTEST_PASSED;

    shrQAStart(argc, argv);

    errors = processArgs (argc, argv, &opts);
    if (errors) {
        fprintf(stdout, "\n Usage: batchcublas [-mSIZE_M] [-nSIZE_N] [-kSIZE_N] [-NSIZE_NUM_ITERATIONS] [-qatest] [-noprompt]\n");
        return CUBLASTEST_FAILED;
    }

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stdout, "CUBLAS initialization failed!\n");
        cudaDeviceReset();
        cutilExit(argc, argv);
    }

    // Run single kernels
    
    fprintf(stdout, "\n ==== Running single kernels ==== \n\n");
    nTimes = opts.N;
    opts.N = 1;
    *(opts.elem_type) = 's';
    status = test_gemm_loop<float>( opts, (float)CUBLAS_SGEMM_MAX_ULP_ERR, (double)CUBLAS_SGEMM_MAX_RELATIVE_ERR, handle );

    // Run Double verion
    *(opts.elem_type) = 'd';
    if (getDeviceVersion() < DEV_VER_DBL_SUPPORT) {
      fprintf(stdout, "@@@@ dgemm test WAIVED due to lack of DP support\n");
      cudaDeviceReset();
      shrQAFinish(argc, (const char **)argv, QA_PASSED);
      exit(EXIT_SUCCESS);
    }            
    status = test_gemm_loop<double>( opts, (float)CUBLAS_DGEMM_MAX_ULP_ERR, (double)CUBLAS_DGEMM_MAX_RELATIVE_ERR, handle );
    nTotalErrors += (status == CUBLASTEST_PASSED ? 0 : 1);
    opts.N = nTimes;

    // Run with and without streams

    for (int ii = 0; ii < 2; ii++) {
        opts.useStream = ii;
        if (ii == 0)
            fprintf(stdout, "\n ==== Running N=%d without streams ==== \n\n", opts.N);
        else
            fprintf(stdout, "\n ==== Running N=%d with streams ==== \n\n", opts.N);

        // Run single version
        *(opts.elem_type) = 's';
        status = test_gemm_loop<float>( opts, (float)CUBLAS_SGEMM_MAX_ULP_ERR, (double)CUBLAS_SGEMM_MAX_RELATIVE_ERR, handle );
        nTotalErrors += (status == CUBLASTEST_PASSED ? 0 : 1);

        // Run Double verion
        *(opts.elem_type) = 'd';

        // Test doesn't meet minSpec, will will wave the DP test
        if (getDeviceVersion() < DEV_VER_DBL_SUPPORT) {
            fprintf(stdout, "@@@@ dgemm test WAIVED due to lack of DP support\n");
            cudaDeviceReset();
            shrQAFinish(argc, (const char **)argv, QA_PASSED);
            exit(EXIT_SUCCESS);
        } else {
            status = test_gemm_loop<double>( opts, (float)CUBLAS_DGEMM_MAX_ULP_ERR, (double)CUBLAS_DGEMM_MAX_RELATIVE_ERR, handle );
            nTotalErrors += (status == CUBLASTEST_PASSED ? 0 : 1);
        }
    }

    cublasDestroy(handle);
    cudaDeviceReset();

    printf("\nTest Summary\n");
    shrQAFinish(argc, (const char **)argv, (nTotalErrors == 0) ? QA_PASSED : QA_FAILED);
}
