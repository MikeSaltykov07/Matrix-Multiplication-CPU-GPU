#include "device.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include < vector      >
#include < time.h      > 
#include < stdio.h     >
#include < cassert     >
#include < cstdlib     >
#include < iostream    >
#include < algorithm   >
#include < functional  >
#include < immintrin.h >
#include < string      >


// CTR+M и CTR+O
#define NUM_STREAM 4
#define BLOCK_SIZE 32
#define MATRIX_SIZE 512 * 16 * 2

#define BASE_TYPE float

#define LOOP_I(_loop) for(int i=0; i < _loop; i++)

using std::vector;
using std::cout;
using std::generate;

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n",
            cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

typedef struct {
    int row = MATRIX_SIZE;
    int col = MATRIX_SIZE;
    float* elements;
    int stride = 2;
} Matrix;

void MatrixMul(float* c, const float* a, const float* b);
int MatrixBlock(float* c, const float* a, const float* b);

void MatrixBlock1(float* c, const float* a, const float* b);

int MatrixBank(float* c, const float* a, const float* b);
void MatrixTiled(float* c, const float* a, const float* b);
void MatrixSub(Matrix C, const Matrix A, const Matrix B);
void MatrixPinned(float* c, const float* a, const float* b);
void MatrixMulStream(float* c, const float* a, const float* b);
void matrixDeviceBuffA(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b);
void matrixDeviceBuffB(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b);
void MatrixMuld(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b);
void matrixSmemm(void smem(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b), BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b, const int stream);



void matrixHost(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b)
{
    for (size_t i = 0; i < MATRIX_SIZE; i++)
    {
        for (size_t j = 0; j < MATRIX_SIZE; j++)
        {
            c[i * MATRIX_SIZE + j] = 0;
            for (size_t k = 0; k < MATRIX_SIZE; k++)
            {
                c[i * MATRIX_SIZE + j] += a[i * MATRIX_SIZE + k] * b[k * MATRIX_SIZE + j];
            }
        }
    }
}


void matrixHostImproved(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b)
{
    for (size_t i = 0; i < MATRIX_SIZE; ++i)
    {
        float* C = c + i * MATRIX_SIZE;
        for (size_t j = 0; j < MATRIX_SIZE; ++j)
        {
            C[j] = 0;
        }
        for (size_t k = 0; k < MATRIX_SIZE; ++k)
        {
            const float* B = b + k * MATRIX_SIZE;
            float A = a[i * MATRIX_SIZE + k];
            for (size_t j = 0; j < MATRIX_SIZE; j++)
            {
                C[j] += A * B[j];
            }
        }
    }
}

void matrixVectorise(float* C, const float* A, float* B)
{
    for (int i = 0; i < MATRIX_SIZE; ++i)
    {
        float* c = C + i * MATRIX_SIZE;
        for (int j = 0; j < MATRIX_SIZE; j += 8)
            _mm256_storeu_ps(c + j + 0, _mm256_setzero_ps());

        for (int k = 0; k < MATRIX_SIZE; ++k)
        {
            const float* b = B + k * MATRIX_SIZE;
            __m256 a = _mm256_set1_ps(A[i * MATRIX_SIZE + k]);
            for (int j = 0; j < MATRIX_SIZE; j += 16)
            {
                _mm256_storeu_ps(c + j + 0, _mm256_fmadd_ps(a, _mm256_loadu_ps(b + j + 0), _mm256_loadu_ps(c + j + 0)));
                _mm256_storeu_ps(c + j + 8, _mm256_fmadd_ps(a, _mm256_loadu_ps(b + j + 8), _mm256_loadu_ps(c + j + 8)));
            }
        }
    }
}

void micro_6x161(int K, const float* A, int lda, int step, const float* B, int ldb, float* C, int ldc)
{
    __m256 c00 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c41 = _mm256_setzero_ps();
    __m256 c51 = _mm256_setzero_ps();
    const int offset0 = lda * 0;
    const int offset1 = lda * 1;
    const int offset2 = lda * 2;
    const int offset3 = lda * 3;
    const int offset4 = lda * 4;
    const int offset5 = lda * 5;
    __m256 b0, b1, b2, a0, a1, a2;
    for (int k = 0; k < K; k++)
    {
        b0 = _mm256_loadu_ps(B + 0);
        b1 = _mm256_loadu_ps(B + 8);
        a0 = _mm256_set1_ps(A[offset0]);
        a1 = _mm256_set1_ps(A[offset1]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        a0 = _mm256_set1_ps(A[offset2]);
        a1 = _mm256_set1_ps(A[offset3]);
        c20 = _mm256_fmadd_ps(a0, b0, c20);
        c21 = _mm256_fmadd_ps(a0, b1, c21);
        c30 = _mm256_fmadd_ps(a1, b0, c30);
        c31 = _mm256_fmadd_ps(a1, b1, c31);
        a0 = _mm256_set1_ps(A[offset4]);
        a1 = _mm256_set1_ps(A[offset5]);
        c40 = _mm256_fmadd_ps(a0, b0, c40);
        c41 = _mm256_fmadd_ps(a0, b1, c41);
        c50 = _mm256_fmadd_ps(a1, b0, c50);
        c51 = _mm256_fmadd_ps(a1, b1, c51);
        B += ldb; A += step;
    }
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c00, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c01, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c10, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c11, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c20, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c21, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c30, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c31, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c40, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c41, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c50, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c51, _mm256_loadu_ps(C + 8)));
}

void init_c1(int M, int N, float* C, int ldc)
{
    for (int i = 0; i < M; ++i, C += ldc)
        for (int j = 0; j < N; j += 8)
            _mm256_storeu_ps(C + j, _mm256_setzero_ps());
}

void matrixCore(BASE_TYPE* C, const BASE_TYPE* A, const BASE_TYPE* B)
{
    for (int i = 0; i < MATRIX_SIZE; i += 6)
    {
        for (int j = 0; j < MATRIX_SIZE; j += 16)
        {
            init_c(6, 16, C + i * MATRIX_SIZE + j, MATRIX_SIZE);
            micro_6x16(MATRIX_SIZE, A + i * MATRIX_SIZE, MATRIX_SIZE, 1, B + j, MATRIX_SIZE, C + i * MATRIX_SIZE + j, MATRIX_SIZE);
        }
    }
}

struct buf_t
{
    float* p;
    int n;

    buf_t(int size) : n(size), p((BASE_TYPE*)_mm_malloc(size * 4, 64)) {}

    ~buf_t() { _mm_free(p); }
};

void reorder_b_161(int K, const float* B, int ldb, float* bufB)
{
    for (int k = 0; k < K; ++k, B += ldb, bufB += 16)
    {
        _mm256_storeu_ps(bufB + 0, _mm256_loadu_ps(B + 0));
        _mm256_storeu_ps(bufB + 8, _mm256_loadu_ps(B + 8));
    }
}

void matrixBuf(float* C,const float* A,const float* B)
{
    for (int j = 0; j < MATRIX_SIZE; j += 16)
    {
        buf_t bufB(16 * MATRIX_SIZE);
        reorder_b_16(MATRIX_SIZE, B + j, MATRIX_SIZE, bufB.p);
        for (int i = 0; i < MATRIX_SIZE; i += 6)
        {
            init_c(6, 16, C + i * MATRIX_SIZE + j, MATRIX_SIZE);
            micro_6x16(MATRIX_SIZE, A + i * MATRIX_SIZE, MATRIX_SIZE, 1, bufB.p, 16, C + i * MATRIX_SIZE + j, MATRIX_SIZE);
        }
    }
}

void macro(BASE_TYPE* C, int K, int ldc, const BASE_TYPE* A, int lda, const BASE_TYPE* B, int ldb, BASE_TYPE* bufB)
{
    for (int j = 0; j < MATRIX_SIZE; j += 16)
    {
        reorder_b_16(K, B + j, ldb, bufB);
        for (int i = 0; i < MATRIX_SIZE; i += 6)
            micro_6x16(K, A + i * lda, lda, 1, bufB, 16, C + i * ldc + j, ldc);
    }
}


void matrixL1(BASE_TYPE* C, const BASE_TYPE* A, const BASE_TYPE* B, int M, int N, int K)
{
    const int L1 = 384 * 1024;
    int mK = std::min(L1 / 4 / 16, K);
    buf_t bufB(16 * mK);
    for (int k = 0; k < K; k += mK)
    {
        int dK = std::min(K, k + mK) - k;
        if (k == 0)
            init_c(M, N, C, N);
        macro(C, dK, N, A + k, K, B + k * N, N, bufB.p);
    }
}

// Без разделения памяти
__global__ void matrixDevice(BASE_TYPE*c, const BASE_TYPE*a, const BASE_TYPE*b)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

    int sum = 0;
    c[i * MATRIX_SIZE/2 + j] = 0;
    for (size_t k = 0; k < MATRIX_SIZE; k++)
    {
        sum += a[i * MATRIX_SIZE/2 + k] * b[j + MATRIX_SIZE * k];
    }
    c[i * MATRIX_SIZE/2 + j] = sum;

}

__global__ void matrixDeviceStream(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b, const int stream)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    int sum = 0;
    c[i * stream + j] = 0;
    for (size_t k = 0; k < MATRIX_SIZE; k++)
    {
        sum += a[i * stream + k] * b[j + MATRIX_SIZE * k];
    }
    c[i * stream + j] = sum;

}


__global__ void matrixDeviceV1(float* c, const float* a, const float* b)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;


    float* C = c + i * MATRIX_SIZE;
    for (size_t m = j; m < j+MATRIX_SIZE; ++m)
    {
        C[m] = 0;
    }
    int sum = 0;
    for (size_t k = 0; k < MATRIX_SIZE; k++)
    {
        const float* B = b + k * MATRIX_SIZE;
        float A = a[i * MATRIX_SIZE + k];
        for (size_t m = j; m < j+MATRIX_SIZE; ++m)
        {
            C[m] += A * B[m];
        }
    }

}


__device__ float GetElement(const Matrix A, int row, int col ) {
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, float value) 
{
    A.elements[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix ASub;
    ASub.row = BLOCK_SIZE;
    ASub.col = BLOCK_SIZE;
    ASub.stride = A.stride;
    ASub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return ASub;
}

__global__ void matrixDeviceSub(Matrix A, Matrix B, Matrix C);
void MatrixMulCublas(float* c, const float* a, const float* b);

// С разделяющей памятью
__global__ void matrixDeviceSub(Matrix C, Matrix A, Matrix B) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    float Cvalue = 0;
    int row = threadIdx.y;
    int col = threadIdx.x;

    for (size_t m = 0; m < (A.row / BLOCK_SIZE); ++m)
    {
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        __syncthreads();

        for (size_t e = 0; e < BLOCK_SIZE; ++e)
        {
            Cvalue += As[row][e] * Bs[e][col];
        }

        __syncthreads();


    }

    SetElement(Csub, row, col, Cvalue);
}

__global__ void matrixDevicBlock(BASE_TYPE* C, BASE_TYPE* A, BASE_TYPE* B)
{
    BASE_TYPE CValue = 0;

    int Row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int Col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ BASE_TYPE As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ BASE_TYPE Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int k = 0; k < (BLOCK_SIZE + MATRIX_SIZE - 1) / BLOCK_SIZE; k++) {

        if (k * BLOCK_SIZE + threadIdx.x < MATRIX_SIZE && Row < MATRIX_SIZE)
            As[threadIdx.y][threadIdx.x] = A[Row * MATRIX_SIZE + k * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if (k * BLOCK_SIZE + threadIdx.y < MATRIX_SIZE && Col < MATRIX_SIZE)
            Bs[threadIdx.y][threadIdx.x] = B[(k * BLOCK_SIZE + threadIdx.y) * MATRIX_SIZE + Col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int n = 0; n < BLOCK_SIZE; ++n)
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

        __syncthreads();
    }

    if (Row < MATRIX_SIZE && Col < MATRIX_SIZE)
        C[((blockIdx.y * blockDim.y + threadIdx.y) * MATRIX_SIZE) + blockIdx.x * blockDim.x + threadIdx.x] = CValue;
}

__global__ void Muld(float* C, const float* A, const float* B)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Index of the first sub-matrix of A processed by the block
    int aBegin = MATRIX_SIZE * BLOCK_SIZE * by;
    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + MATRIX_SIZE - 1;
    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;
    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;
    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * MATRIX_SIZE;
    // The element of the block sub-matrix that is computed
    // by the thread
    float Csub = 0;
    // Loop over all the sub-matrices of A and B required to
    // compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
        a <= aEnd;
        a += aStep, b += bStep) {
        // Shared memory for the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        // Shared memory for the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load the matrices from global memory to shared memory;
        // each thread loads one element of each matrix
        As[ty][tx] = A[a + MATRIX_SIZE * ty + tx];
        Bs[ty][tx] = B[b + MATRIX_SIZE * ty + tx];
        // Synchronize to make sure the matrices are loaded
        __syncthreads();
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write the block sub-matrix to global memory;
    // each thread writes one element
    int c = MATRIX_SIZE * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + MATRIX_SIZE * ty + tx] = Csub;
}


__global__ void matrixPin(float* __restrict c, const float* __restrict a, const float* __restrict b, int N) {
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    c[row * N + col] = 0;
    for (int k = 0; k < N; k++) {
        // Accumulate results for a single element
        c[row * N + col] += a[row * N + k] * b[k * N + col];
    }
}

__global__ void function(float* dA, float* dB, float* dC, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) dC[i] = dA[i] + dB[i];
}


// Pull out matrix and shared memory tile size 
const int N = 1024;
const int SHMEM_SIZE = 1024;

// Shared memory bank conflicts
__global__ void matrixMultBank(BASE_TYPE* C, const BASE_TYPE* A, const BASE_TYPE* B)
{
    // индекс начала первой подматрицы А, которую
    // обрабатывает блок
    int aBegin = MATRIX_SIZE * blockDim.y * blockIdx.y;
    // индекс конца подматрицы А, которую обрабатывает блок
    int aEnd = aBegin + MATRIX_SIZE - 1;
    // шаг для перебора подматриц А
    int aStep = blockDim.x;
    // индекс начала первой подматрицы В, которую
    // обрабатывает блок
    int bBegin = blockDim.x * blockIdx.x;
    // шаг для перебора подматриц В
    int bStep = blockDim.y * MATRIX_SIZE;

    // Выделение разделяемой памяти для подматриц
    __shared__ BASE_TYPE as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ BASE_TYPE bs[BLOCK_SIZE][BLOCK_SIZE];

    // переменная для вычисления элемента подматрицы
    BASE_TYPE sum = 0.0;
    for (int ia = aBegin, ib = bBegin; ia < aEnd; ia +=
        aStep, ib += bStep)
    {
        // загрузка подматриц А и В из глобальной памяти в
        // разделяемую
        as[threadIdx.y][threadIdx.x] = A[ia + MATRIX_SIZE * threadIdx.y + threadIdx.x];
        bs[threadIdx.y][threadIdx.x] = B[ib + MATRIX_SIZE * threadIdx.y + threadIdx.x];
        // синхронизация нитей
        __syncthreads();
        // перемножение двух матриц
        for (int k = 0; k < blockDim.x; k++)
            sum += as[threadIdx.y][k] *
            bs[k][threadIdx.x];
        // синхронизация нитей
        __syncthreads();
    }
    // индекс результирующего элемента в глобальной памяти
    int ind = MATRIX_SIZE * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    // запись элемента в глобальную память
    C[ind] = sum;
}

__global__ void matrixSmem1(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int aBegin = MATRIX_SIZE * BLOCK_SIZE * by, aEnd = aBegin + MATRIX_SIZE - 1;
    int bBegin = BLOCK_SIZE * bx, aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * MATRIX_SIZE;
    BASE_TYPE sum = 0.0f;
    __shared__ BASE_TYPE as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ BASE_TYPE bs[BLOCK_SIZE][BLOCK_SIZE];
    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
    {
        as[tx][ty] = a[ia + MATRIX_SIZE * ty + tx];
        bs[tx][ty] = b[ib + MATRIX_SIZE * ty + tx];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) sum += as[k][ty] * bs[tx][k];
        __syncthreads();
    }
    c[aBegin + bBegin + ty * MATRIX_SIZE + tx] = sum;
}
__global__ void matrixSmem2(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int aBegin = MATRIX_SIZE * BLOCK_SIZE * by, aEnd = aBegin + MATRIX_SIZE - 1;
    int bBegin = BLOCK_SIZE * bx, aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * MATRIX_SIZE;
    BASE_TYPE sum = 0.0f;
    __shared__ BASE_TYPE as[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ BASE_TYPE bs[BLOCK_SIZE][BLOCK_SIZE + 1];
    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
    {
        as[tx][ty] = a[ia + MATRIX_SIZE * ty + tx];
        bs[tx][ty] = b[ib + MATRIX_SIZE * ty + tx];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) sum += as[k][ty] * bs[tx][k];
        __syncthreads();
    }
    c[aBegin + bBegin + ty * MATRIX_SIZE + tx] = sum;
}
__global__ void matrixSmem3(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int aBegin = MATRIX_SIZE * BLOCK_SIZE * by, aEnd = aBegin + MATRIX_SIZE - 1;
    int bBegin = BLOCK_SIZE * bx, aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * MATRIX_SIZE;
    BASE_TYPE sum = 0.0f;
    __shared__ BASE_TYPE as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ BASE_TYPE bs[BLOCK_SIZE][BLOCK_SIZE];
    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
    {
        as[ty][tx] = a[ia + MATRIX_SIZE * ty + tx];
        bs[ty][tx] = b[ib + MATRIX_SIZE * ty + tx];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += as[ty][k] * bs[k][tx];
        __syncthreads();
    }
    c[aBegin + bBegin + ty * MATRIX_SIZE + tx] = sum;
}
__global__ void matrixSmem4(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int aBegin = MATRIX_SIZE * BLOCK_SIZE * by, aEnd = aBegin + MATRIX_SIZE - 1;
    int bBegin = BLOCK_SIZE * bx, aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * MATRIX_SIZE;
    BASE_TYPE sum1 = 0.0f, sum2 = 0.0f;
    __shared__ BASE_TYPE as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ BASE_TYPE bs[BLOCK_SIZE][BLOCK_SIZE];
    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
    {
        as[ty][tx] = a[ia + MATRIX_SIZE * ty + tx];
        bs[ty][tx] = b[ib + MATRIX_SIZE * ty + tx];
        as[ty + 16][tx] = a[ia + MATRIX_SIZE * (ty + 16) + tx];
        bs[ty + 16][tx] = b[ib + MATRIX_SIZE * (ty + 16) + tx];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum1 += as[ty][k] * bs[k][tx];
            sum2 += as[ty + 16][k] * bs[k][tx];
        }
        __syncthreads();
    }
    c[aBegin + bBegin + ty * MATRIX_SIZE + tx] = sum1;
    c[aBegin + bBegin + (ty + 16) * MATRIX_SIZE + tx] = sum2;
}
__global__ void matrixSmem5(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int aBegin = MATRIX_SIZE * BLOCK_SIZE * by, aEnd = aBegin + MATRIX_SIZE - 1;
    int bBegin = BLOCK_SIZE * bx, aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * MATRIX_SIZE;
    BASE_TYPE sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
    __shared__ BASE_TYPE as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ BASE_TYPE bs[BLOCK_SIZE][BLOCK_SIZE];
    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep)
    {
        as[ty][tx] = a[ia + MATRIX_SIZE * ty + tx];
        bs[ty][tx] = b[ib + MATRIX_SIZE * ty + tx];
        as[ty + 8][tx] = a[ia + MATRIX_SIZE * (ty + 8) + tx];
        bs[ty + 8][tx] = b[ib + MATRIX_SIZE * (ty + 8) + tx];
        as[ty + 16][tx] = a[ia + MATRIX_SIZE * (ty + 16) + tx];
        bs[ty + 16][tx] = b[ib + MATRIX_SIZE * (ty + 16) + tx];
        as[ty + 24][tx] = a[ia + MATRIX_SIZE * (ty + 24) + tx];
        bs[ty + 24][tx] = b[ib + MATRIX_SIZE * (ty + 24) + tx];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum1 += as[ty][k] * bs[k][tx];
            sum2 += as[ty + 8][k] * bs[k][tx];
            sum3 += as[ty + 16][k] * bs[k][tx];
            sum4 += as[ty + 24][k] * bs[k][tx];
        }
        __syncthreads();
    }
    c[aBegin + bBegin + ty * MATRIX_SIZE + tx] = sum1;
    c[aBegin + bBegin + (ty + 8) * MATRIX_SIZE + tx] = sum2;
    c[aBegin + bBegin + (ty + 16) * MATRIX_SIZE + tx] = sum3;
    c[aBegin + bBegin + (ty + 24) * MATRIX_SIZE + tx] = sum4;
}


__global__ void vectorAdd(float* c, const float* a, const float* b, int N) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Boundary check
    if (tid < N) {
        // Each thread adds a single element
        c[tid] = a[tid] + b[tid];
    }
}


__global__ void kernel_global(float* c, const float* a, const float* b)
{
    int bx = blockIdx.x; // номер блока по x
    int by = blockIdx.y; // номер блока по y
    int tx = threadIdx.x; // номер нити в блоке по x
    int ty = threadIdx.y; // номер нити в блоке по y
    float sum = 0.0f;
    int ia = MATRIX_SIZE * (BLOCK_SIZE * by + ty); // номер строки из A’
    int ib = BLOCK_SIZE * bx + tx; // номер столбца из B’
    int ic = ia + ib; // номер элемента из С’
    // вычисление элемента матрицы C
    for (int k = 0; k < MATRIX_SIZE; k++) sum += a[ia + k] * b[ib + k * MATRIX_SIZE];
    c[ic] = sum;
}


__global__ void matrixMulTiled(BASE_TYPE* c, const BASE_TYPE* a,const BASE_TYPE* b) {
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Statically allocated shared memory
    __shared__ BASE_TYPE s_a[SHMEM_SIZE];
    __shared__ BASE_TYPE s_b[SHMEM_SIZE];

    // Accumulate in temporary variable
    int tmp = 0;

    // Sweep tile across matrix
    for (int i = 0; i < MATRIX_SIZE; i += blockDim.x) {
        // Load in elements for this tile
        s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * MATRIX_SIZE + i + threadIdx.x];
        s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * MATRIX_SIZE + threadIdx.y * MATRIX_SIZE + col];

        // Wait for both tiles to be loaded in before doing computation
        __syncthreads();

        // Do matrix multiplication on the small matrix
        for (int j = 0; j < blockDim.x; j++) {
            tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
        }

        // Wait for all threads to finish using current tiles before loading in new
        // ones
        __syncthreads();
    }

    // Write back results
    c[row * MATRIX_SIZE + col] = tmp;
}

double experiment(void function(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b), 
        const std::string type, const std::string description, BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b)
{
    const int testCount = 10;
    double seconds;
    clock_t start, end;

    LOOP_I(testCount) {

        start = clock();

        function(c, a, b);

        end = clock();

        seconds += (double)(end - start) / CLOCKS_PER_SEC;

    }
    for (size_t i = 200; i < 210; i++)
    {
        std::cout << c[i] << " ";
    }

    c = new float[MATRIX_SIZE * MATRIX_SIZE];
    printf("time %s: %.2f ms - %s\n", type, seconds / testCount, description);

    return seconds;
}

const int TRX = 16;
const int TRY = 16;

struct gpu_buf_t
{
    float* p;
    int n;

    gpu_buf_t(int size)
        : n(size)
        , p(0)
    {
        cudaError_t error = cudaMalloc(&p, n * sizeof(float));
        assert(error == cudaSuccess);
    }

    ~gpu_buf_t()
    {
        if (p)
        {
            cudaError_t error = cudaFree(p);
            assert(error == cudaSuccess);
            p = 0;
        }
    }
};

const int TSM = 128;
const int TSN = 128;
const int TSK = 16;
const int WPTM = 8;
const int WPTN = 8;
const int RTSM = TSM / WPTM;
const int RTSN = TSN / WPTN;
const int LPTA = TSK * WPTM * WPTN / TSN;

__global__ void transpose(int P, int Q, const float* src, float* dst)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    const int ID0 = blockIdx.x * TRX + tx;
    const int ID1 = blockIdx.y * TRY + ty;
    __shared__ float buf[TRX][TRY];
    if (ID0 < P && ID1 < Q)
        buf[ty][tx] = src[ID1 * P + ID0];
    __syncthreads();
    const int newID0 = blockIdx.y * TRY + tx;
    const int newID1 = blockIdx.x * TRX + ty;
    if (newID0 < Q && newID1 < P)
        dst[newID1 * Q + newID0] = buf[tx][ty];
}

__global__ void gemm(int M, int N, int K, const float* A, const float* B, float* C)
{
    const int tidm = threadIdx.y;
    const int tidn = threadIdx.x;
    const int offsetM = TSM * blockIdx.y;
    const int offsetN = TSN * blockIdx.x;

    __shared__ float sA[TSK][TSM];
    __shared__ float sB[TSN][TSK];

    float rA;
    float rB[WPTN];
    float rC[WPTM][WPTN];

#pragma unroll
    for (int wm = 0; wm < WPTM; wm++)
    {
#pragma unroll
        for (int wn = 0; wn < WPTN; wn++)
            rC[wm][wn] = 0.0f;
    }

    for (int k0 = 0; k0 < K; k0 += TSK)
    {
#pragma unroll
        for (int la = 0; la < LPTA; la++)
        {
            int tid = tidn * RTSM + tidm;
            int id = la * RTSN * RTSM + tid;
            int row = id % TSM;
            int col = id / TSM;
            int tiledIndex = k0 + col;
#if __CUDA_ARCH__ >= 320
            sA[col][row] = __ldg(&A[tiledIndex * M + offsetM + row]);
            sB[row][col] = __ldg(&B[tiledIndex * N + offsetN + row]);
#else
            sA[col][row] = A[tiledIndex * M + offsetM + row];
            sB[row][col] = B[tiledIndex * N + offsetN + row];
#endif
        }
        __syncthreads();
        for (int k = 0; k < TSK; k++)
        {
#pragma unroll
            for (int wn = 0; wn < WPTN; wn++)
            {
                int col = tidn + wn * RTSN;
                rB[wn] = sB[col][k];
            }
#pragma unroll
            for (int wm = 0; wm < WPTM; wm++)
            {
                int row = tidm + wm * RTSM;
                rA = sA[k][row];
#pragma unroll
                for (int wn = 0; wn < WPTN; wn++) {
                    rC[wm][wn] += rA * rB[wn];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int wm = 0; wm < WPTM; wm++)
    {
        int globalRow = offsetM + tidm + wm * RTSM;
#pragma unroll
        for (int wn = 0; wn < WPTN; wn++)
        {
            int globalCol = offsetN + tidn + wn * RTSN;
            C[globalCol + globalRow * N] = rC[wm][wn];
        }
    }
}


const int TS = 32;
const int WPT = 8;
const int PTS = TS / WPT;

__global__ void transposeBuffB(int P, int Q, const float* src, float* dst)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    const int ID0 = blockIdx.x * TRX + tx;
    const int ID1 = blockIdx.y * TRY + ty;
    __shared__ float buf[TRX][TRY];
    if (ID0 < P && ID1 < Q)
        buf[ty][tx] = src[ID1 * P + ID0];
    __syncthreads();
    const int newID0 = blockIdx.y * TRY + tx;
    const int newID1 = blockIdx.x * TRX + ty;
    if (newID0 < Q && newID1 < P)
        dst[newID1 * Q + newID0] = buf[tx][ty];
}

__global__ void MatrixBuffB(int M, int N, int K, const float* A, const float* B, float* C)
{
    int i0 = TS * blockIdx.y + threadIdx.y;
    int j = TS * blockIdx.x + threadIdx.x;
    float c[WPT];
    for (int w = 0; w < WPT; w++)
        c[w] = 0.0f;
    __shared__ float sA[TS][TS];
    __shared__ float sB[TS][TS];
    for (int k0 = 0; k0 < K; k0 += TS)
    {
        for (int w = 0; w < WPT; w++)
        {
            sA[threadIdx.y + w * PTS][threadIdx.x] = A[(i0 + w * PTS) * K + (k0 + threadIdx.x)];
            sB[threadIdx.y + w * PTS][threadIdx.x] = B[(j)*K + (k0 + threadIdx.y + w * PTS)];
        }
        __syncthreads();
        for (int k = 0; k < TS; ++k)
        {
            float b = sB[k][threadIdx.x];
            for (int w = 0; w < WPT; w++)
                c[w] += sA[threadIdx.y + w * PTS][k] * b;
        }
        __syncthreads();
    }
    for (int w = 0; w < WPT; w++)
        C[(i0 + w * PTS) * N + j] = c[w];
}


int main()
{
    double seconds;
    double t1;
    double t2;
    double t;
    clock_t start, end;
    Matrix d;
    unsigned int mem_size = sizeof(float) * MATRIX_SIZE * MATRIX_SIZE;

    float* a = new float[MATRIX_SIZE * MATRIX_SIZE];
    float* b = new float[MATRIX_SIZE * MATRIX_SIZE];
    float* c = new float[MATRIX_SIZE * MATRIX_SIZE];

    for (size_t i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++)
    {
        a[i] = rand() % 5;
        b[i] = rand() % 5;
    }

    // TEST

    {
        // Standart 
        experiment(MatrixMul, "GPU", "Standart", c, a, b);

        cudaDeviceReset();

        //
        // Cuda standart + pinned
        //
        {
            float* a_pin, * b_pin, * c_pin;
            unsigned int mem_size = sizeof(float) * MATRIX_SIZE * MATRIX_SIZE;

            cudaHostAlloc((void**)&a_pin, mem_size, cudaHostAllocDefault);
            cudaHostAlloc((void**)&b_pin, mem_size, cudaHostAllocDefault);
            cudaHostAlloc((void**)&c_pin, mem_size, cudaHostAllocDefault);


            for (size_t i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++)
            {
                a_pin[i] = a[i];
                b_pin[i] = b[i];
            }

            experiment(MatrixMul, "GPU", "Standart+Pinned", c_pin, a_pin, b_pin);

            cudaDeviceReset();
            cudaFreeHost(a_pin);
            cudaFreeHost(b_pin);
            cudaFreeHost(c_pin);
        }
        // Cuda Pinned + Stream
        {
            float* a_pin, * b_pin, * c_pin;
            unsigned int mem_size = sizeof(float) * MATRIX_SIZE * MATRIX_SIZE;
            cudaHostAlloc((void**)&a_pin, mem_size, cudaHostAllocDefault);
            cudaHostAlloc((void**)&b_pin, mem_size, cudaHostAllocDefault);
            cudaHostAlloc((void**)&c_pin, mem_size, cudaHostAllocDefault);

            for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
                a_pin[i] = a[i];
                b_pin[i] = b[i];
            }


            experiment(MatrixMulStream, "GPU", "Standart+Stream", c_pin, a_pin, b_pin);

            cudaDeviceReset();
            cudaFreeHost(a_pin);
            cudaFreeHost(b_pin);
            cudaFreeHost(c_pin);
        }
        //

        start = clock();

        matrixSmemm(matrixSmem1,c, a, b, 1);

        end = clock();

        for (size_t i = 200; i < 210; i++)
        {
            std::cout << c[i] << " ";
        }

        seconds = (double)(end - start) / CLOCKS_PER_SEC;
        printf("time GPU: %.2f ms - Smem1 \n", seconds);

        cudaDeviceReset();

        start = clock();

        matrixSmemm(matrixSmem2, c, a, b, 1);

        end = clock();

        for (size_t i = 200; i < 210; i++)
        {
            std::cout << c[i] << " ";
        }

        seconds = (double)(end - start) / CLOCKS_PER_SEC;
        printf("time GPU: %.2f ms - Smem2 \n", seconds);

        cudaDeviceReset();

        start = clock();

        matrixSmemm(matrixSmem3, c, a, b, 1);

        end = clock();

        for (size_t i = 200; i < 210; i++)
        {
            std::cout << c[i] << " ";
        }

        seconds = (double)(end - start) / CLOCKS_PER_SEC;
        printf("time GPU: %.2f ms - Smem3 \n", seconds);

        cudaDeviceReset();

        start = clock();

        matrixSmemm(matrixSmem4, c, a, b, 2);

        end = clock();

        for (size_t i = 200; i < 210; i++)
        {
            std::cout << c[i] << " ";
        }

        seconds = (double)(end - start) / CLOCKS_PER_SEC;
        printf("time GPU: %.2f ms - Smem4 \n", seconds);

        cudaDeviceReset();

        start = clock();

        matrixSmemm(matrixSmem5, c, a, b, 4);

        end = clock();

        for (size_t i = 200; i < 210; i++)
        {
            std::cout << c[i] << " ";
        }

        seconds = (double)(end - start) / CLOCKS_PER_SEC;
        printf("time GPU: %.2f ms - Smem5 \n", seconds);

        cudaDeviceReset();

        experiment(MatrixMuld, "GPU", "MatrixMuldSub", c, a, b);

        cudaDeviceReset();

        experiment(matrixDeviceBuffA, "GPU", "BuffA",c, a, b);

        cudaDeviceReset();

        experiment(matrixDeviceBuffB, "GPU", "BuffB", c, a, b);

        cudaDeviceReset();

        experiment(MatrixBlock1, "GPU", "Block Mult", c, a, b);

        cudaDeviceReset();

        experiment(MatrixTiled, "GPU", "Tiled Mult", c, a, b);

        cudaDeviceReset();


    }

    // Базовый
    

    //experiment(matrixHost, "CPU", "Standart", c, a, b

    // gemm
    //experiment(matrixDevice, "GPU", "gemm2", c, a, b);

    //{


    //start = clock();

    //gemm_v2(MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, a, b, c);

    //end = clock();

    //for (size_t i = 200; i < 210; i++)
    //{
    //std::cout << c[i] << " ";
    //}
    //t1 = (double)(end - start) / CLOCKS_PER_SEC;
    //printf("time CPU: %.2f ms - SIMD\n", t1);


    //c = new float[MATRIX_SIZE * MATRIX_SIZE];

    //start = clock();

    //gemm_v3(MATRIX_SIZE - 6, MATRIX_SIZE, MATRIX_SIZE, a, b, c);

    //end = clock();

    //for (size_t i = 200; i < 210; i++)
    //{
    //std::cout << c[i] << " ";
    //}

    //t2 = (double)(end - start) / CLOCKS_PER_SEC;
    //printf("time CPU: %.2f ms - Core\n", t2);

    //c = new float[MATRIX_SIZE * MATRIX_SIZE];

    //start = clock();

    //gemm_v4(MATRIX_SIZE - 6, MATRIX_SIZE, MATRIX_SIZE, a, b, c);

    //end = clock();

    //for (size_t i = 200; i < 210; i++)
    //{
    //std::cout << c[i] << " ";
    //}

    //seconds = (double)(end - start) / CLOCKS_PER_SEC;
    //printf("time CPU: %.2f ms - BuffB\n", seconds);

    //c = new float[MATRIX_SIZE * MATRIX_SIZE];

    //start = clock();

    //gemm_v5(MATRIX_SIZE - 6, MATRIX_SIZE, MATRIX_SIZE, a, b, c);

    //end = clock();

    //for (size_t i = 200; i < 210; i++)
    //{
    //std::cout << c[i] << " ";
    //}

    //seconds = (double)(end - start) / CLOCKS_PER_SEC;
    //printf("time CPU: %.2f ms - L1 \n", seconds);

    //c = new float[MATRIX_SIZE * MATRIX_SIZE];

    //start = clock();

    //gemm_v6(MATRIX_SIZE - 6, MATRIX_SIZE, MATRIX_SIZE, a, b, c);

    //end = clock();

    //for (size_t i = 200; i < 210; i++)
    //{
    //std::cout << c[i] << " ";
    //}

    //seconds = (double)(end - start) / CLOCKS_PER_SEC;
    //printf("time CPU: %.2f ms - L2 \n\n", seconds);

    //c = new float[MATRIX_SIZE * MATRIX_SIZE];

    //start = clock();

    //gemm_v6(MATRIX_SIZE - 6, MATRIX_SIZE, MATRIX_SIZE, a, b, c);

    //end = clock();

    //for (size_t i = 200; i < 210; i++)
    //{
    //std::cout << c[i] << " ";
    //}

    //seconds = (double)(end - start) / CLOCKS_PER_SEC;
    //printf("time CPU: %.2f ms - L3\n\n", seconds);
    //}

    //
    // Cuda Standart
    //
    {




   // for (size_t i = 200; i < 210; i++)
    //{
    //    std::cout << c[i] << " ";
    //}

    //t = (double)(end - start) / CLOCKS_PER_SEC;
    //printf("time GPU: %.2f standart\n", t);
    //printf("GPU vs CPU:  %.2f\n", t1 + t);
    //printf("GPU vs CPU:  %.2f\n", t2);

    }
    


    //cudaDeviceReset();
    //
    //experiment(MatrixBank, "GPU", "Bank Mult", c, a, b);

    //cudaDeviceReset();

    //printf("\n----------------------experiment-----------------------\n");

    //experiment(MatrixBlock, "GPU", "Block Mult", c, a, b);

    //cudaDeviceReset();
    //
    // Cuda BLock
    //{
    //start = clock();

    //MatrixBlock1(c, a, b);

    //end = clock();

    //for (size_t i = 200; i < 210; i++)
    //{
    //    std::cout << c[i] << " ";
    //}

    //t = (double)(end - start) / CLOCKS_PER_SEC;
    //printf("time GPU: %.2f ms - Block Mult\n", t);
    //cudaDeviceReset();
    //}
    ////printf("----------------------experiment-----------------------\n\n");

    ////
    //// Cuda Sub
    ////
    //{
    //    c = new float[MATRIX_SIZE * MATRIX_SIZE];

    //    int numBytes = sizeof(float) * MATRIX_SIZE * MATRIX_SIZE;
    //    Matrix A, B, C;
    //    A.elements = a;
    //    B.elements = b;
    //    C.elements = c;


    //    start = clock();

    //    MatrixSub(C, A, B);

    //    end = clock();

    //    for (size_t i = 200; i < 210; i++)
    //    {
    //        std::cout << C.elements[i] << " ";
    //    }

    //    seconds = (double)(end - start) / CLOCKS_PER_SEC;
    //    printf("time GPU: %.2f ms - Sub Mult\n", seconds);

    //    cudaDeviceReset();

    //    c = new float[MATRIX_SIZE * MATRIX_SIZE];
    //}

    //
    //// Cuda Pinnet
    //
    //{
    //    start = clock();

    //    MatrixPinned(c, a, b);

    //    end = clock();

    //    for (size_t i = 200; i < 210; i++)
    //    {
    //        std::cout << c[i] << " ";
    //    }

    //    seconds = (double)(end - start) / CLOCKS_PER_SEC;
    //    //printf("time GPU: %.2f ms - Pinned\n", seconds);
    //    cudaDeviceReset();
    //}
    //{
    //    float timerValueGPU, timerValueCPU;
    //    cudaEvent_t start, stop;
    //    cudaEventCreate(&start);
    //    cudaEventCreate(&stop);
    //    float* hA, * hB, * hC, * dA, * dB, * dC;
    //    int size = MATRIX_SIZE * MATRIX_SIZE; // размер каждого массива
    //    int N_thread = MATRIX_SIZE; // число нитей в блоке
    //    int N_blocks, i;

    //    // задание массивов hA,hB,hC для host
    //    unsigned int mem_size = sizeof(float) * size;
    //    hA = (float*)malloc(mem_size);
    //    hB = (float*)malloc(mem_size);
    //    hC = (float*)malloc(mem_size);
    //    // выделение памяти на device под массивы hA, hB, hC
    //    cudaMalloc((void**)&dA, mem_size);
    //    cudaMalloc((void**)&dB, mem_size);
    //    cudaMalloc((void**)&dC, mem_size);
    //    // заполнение массивов hA,hB и обнуление hC
    //    for (i = 0; i < size; i++)
    //    {
    //        hA[i] = a[i];
    //        hB[i] = b[i];
    //        hC[i] = 0.0f;
    //    }


    //    // определение числа блоков
    //    if ((size % N_thread) == 0)
    //    {
    //        N_blocks = size / N_thread;
    //    }
    //    else
    //    {
    //        N_blocks = (int)(size / N_thread) + 1;
    //    }
    //    dim3 blocks(N_blocks);
    //    dim3 dimBLock(BLOCK_SIZE, BLOCK_SIZE);
    //    dim3 dimGrid(MATRIX_SIZE / dimBLock.x, MATRIX_SIZE / dimBLock.y);
    //    // ----------------------GPU вариант -------------------
    //    // Старт таймера
    //    cudaEventRecord(start, 0);
    //    // Копирование массивов с host на device
    //    cudaMemcpy(dA, a, mem_size, cudaMemcpyHostToDevice);
    //    cudaMemcpy(dB, b, mem_size, cudaMemcpyHostToDevice);
    //    // Запуск функции-ядра
    //    function << < dimBLock, dimGrid >> > (dA, dB, dC, size);

    //    // Копирование результат с device на host
    //    cudaMemcpy(hC, dC, mem_size, cudaMemcpyDeviceToHost);
    //    for (size_t i = 200; i < 210; i++)
    //    {
    //        std::cout << ":" << hC[i] << " ";
    //    }
    //    // Остановка таймера и вывод времени
    //    // вычисления GPU варианта
    //    cudaEventRecord(stop, 0);
    //    cudaEventSynchronize(stop);
    //    cudaEventElapsedTime(&timerValueGPU, start, stop);
    //    printf("\n GPU calculation time: %f ms\n", timerValueGPU);
    //}



    ////
    //// Cublas
    ////
    //{
    //    start = clock();

    //    MatrixMulCublas(c, a, b);

    //    end = clock();

    //    for (size_t i = 200; i < 210; i++)
    //    {
    //        std::cout << c[i] << " ";
    //    }

    //    seconds = (double)(end - start) / CLOCKS_PER_SEC;
    //    printf("time GPU: %.2f ms - Cublas \n", seconds);

    //    cudaDeviceReset();
    //}
    //

    delete a;
    delete b;
    delete c;

    return 0;
}


void matrixSmemm(void smem(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b), BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b, const int stream)
{


    int numBytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(BASE_TYPE);

    BASE_TYPE* dev_a = 0;
    BASE_TYPE* dev_b = 0;
    BASE_TYPE* dev_c = 0;



    checkCuda(cudaSetDevice(0));

    checkCuda(cudaMalloc((void**)&dev_c, numBytes));
    checkCuda(cudaMalloc((void**)&dev_a, numBytes));
    checkCuda(cudaMalloc((void**)&dev_b, numBytes));

    checkCuda(cudaMemcpy(dev_a, a, numBytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_b, b, numBytes, cudaMemcpyHostToDevice));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE / stream);
    dim3 blocks(N / threads.x, N / threads.y);

    smem << <blocks, threads >> > (dev_c, dev_a, dev_b);

    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaGetLastError());


    checkCuda(cudaMemcpy(c, dev_c, numBytes, cudaMemcpyDeviceToHost));

    checkCuda(cudaFree(dev_c));
    checkCuda(cudaFree(dev_a));
    checkCuda(cudaFree(dev_b));
}

void MatrixPinned(float* c, const float* a, const float* b)
{
    int numBytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    // 2^16
    constexpr int N = MATRIX_SIZE;
    size_t bytes = numBytes;
    unsigned dataArraySize = MATRIX_SIZE * MATRIX_SIZE / sizeof(float);
    //vector <string> ivector;


    vector<int> h_a(MATRIX_SIZE * MATRIX_SIZE);
    vector<int> h_b(MATRIX_SIZE * MATRIX_SIZE);
    vector<int> h_c(MATRIX_SIZE * MATRIX_SIZE);

    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        h_a[i] = a[i];
        h_b[i] = b[i];
    }

    //h_a.insert(h_a.end(), &a[0], &a[numBytes]);
    //h_b.insert(h_b.end(), &b[0], &b[numBytes]);

    ////// Initialize matrices
    //std: generate(h_a.begin(), h_a.end(), a);
    //generate(h_b.begin(), h_b.end(), b);

    // // Copy data to the device
    //cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // Vectors for holding the host-side (CPU-side) data
    //float* h_a, * h_b, * h_c;

    // Allocate pinned memory
    //cudaMallocHost(&h_a, bytes);
    //cudaMallocHost(&h_b, bytes);
    //cudaMallocHost(&h_c, bytes);

    // Threads per CTA(1024 threads per CTA)

    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;

    //dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    //dim3 grid(MATRIX_SIZE / block.x, MATRIX_SIZE / block.y);
     // Threads per CTA dimension
    int THREADS = 32;

    // Blocks per grid dimension (assumes THREADS divides N evenly)
    int BLOCKS = N / THREADS;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    checkCuda(cudaSetDevice(0));

    checkCuda(cudaMalloc((void**)&dev_c, numBytes));
    checkCuda(cudaMalloc((void**)&dev_a, numBytes));
    checkCuda(cudaMalloc((void**)&dev_b, numBytes));


    checkCuda(cudaMemcpy(dev_a, h_a.data(), numBytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_b, h_b.data(), numBytes, cudaMemcpyHostToDevice));

    matrixDevice << <blocks, threads >> > (dev_c, dev_a, dev_b);

    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaGetLastError());

    checkCuda(cudaMemcpy(h_c.data(), dev_c, numBytes, cudaMemcpyDeviceToHost));
    //for (auto x : h_c)
    //    std::cout << x << ' ';

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}

void MatrixSub(Matrix C, const Matrix A, const Matrix B) 
{
    size_t numBytes = A.col * A.row * sizeof(float);

    Matrix dev_a;
    Matrix dev_b;
    Matrix dev_c;

    // Load A and B to device memory
    dev_a.col = dev_a.stride = A.col; dev_a.row = A.row;
    checkCuda(  cudaMalloc((void**)&dev_a.elements, numBytes))  ;
    checkCuda(  cudaMemcpy(dev_a.elements, A.elements, numBytes, cudaMemcpyHostToDevice)  );

    dev_b.col = dev_b.stride = A.col; dev_b.row = B.row;
    checkCuda(  cudaMalloc((void**)&dev_b.elements, numBytes)  );
    checkCuda(  cudaMemcpy(dev_b.elements, B.elements, numBytes, cudaMemcpyHostToDevice)  );

    dev_c.col = dev_c.stride = C.col; dev_c.row = C.row;
    checkCuda(  cudaMalloc((void**)&dev_c.elements, numBytes)  );
    checkCuda(  cudaMemcpy(dev_c.elements, C.elements, numBytes, cudaMemcpyHostToDevice)  );

    dim3 dimBLock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid (MATRIX_SIZE / dimBLock.x, MATRIX_SIZE / dimBLock.y);
    matrixDeviceSub <<<dimGrid, dimBLock>>> (dev_c, dev_a, dev_b);

    checkCuda(  cudaGetLastError()  );

    checkCuda(  cudaDeviceSynchronize()  );

    checkCuda(  cudaMemcpy(C.elements, dev_c.elements, numBytes, cudaMemcpyDeviceToHost)  );

    cudaFree(dev_c.elements);
    cudaFree(dev_a.elements);
    cudaFree(dev_b.elements);

}

int MatrixBlock(float* c, const float* a, const float* b)
{
    int numBytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(MATRIX_SIZE / block.x, MATRIX_SIZE / block.y);
    
    checkCuda(  cudaSetDevice(0)  );
    
    checkCuda(  cudaMalloc((void**)&dev_c, numBytes)  );
    checkCuda(  cudaMalloc((void**)&dev_a, numBytes)  );
    checkCuda(  cudaMalloc((void**)&dev_b, numBytes)  );
    
    checkCuda(  cudaMemcpy(dev_a, a, numBytes, cudaMemcpyHostToDevice)  );
    checkCuda(  cudaMemcpy(dev_b, b, numBytes, cudaMemcpyHostToDevice)  );

    matrixDevicBlock <<<grid, block>>> (dev_c, dev_a, dev_b);

    checkCuda(  cudaDeviceSynchronize()  );

    checkCuda(  cudaGetLastError()  );
    
    checkCuda(  cudaMemcpy(c, dev_c, numBytes, cudaMemcpyDeviceToHost) );

    cudaFree(   dev_c   );
    cudaFree(   dev_a   );
    cudaFree(   dev_b   );

    return 0;
}

void MatrixBlock1(float* c, const float* a, const float* b)
{
    int numBytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(MATRIX_SIZE / block.x, MATRIX_SIZE / block.y);

    checkCuda(cudaSetDevice(0));

    checkCuda(cudaMalloc((void**)&dev_c, numBytes));
    checkCuda(cudaMalloc((void**)&dev_a, numBytes));
    checkCuda(cudaMalloc((void**)&dev_b, numBytes));

    checkCuda(cudaMemcpy(dev_a, a, numBytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_b, b, numBytes, cudaMemcpyHostToDevice));

    matrixDevicBlock << <grid, block >> > (dev_c, dev_a, dev_b);

    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaGetLastError());

    checkCuda(cudaMemcpy(c, dev_c, numBytes, cudaMemcpyDeviceToHost));

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}

void MatrixTiled(float* c, const float* a, const float* b)
{
    int numBytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(MATRIX_SIZE / block.x, MATRIX_SIZE / block.y);
    
    checkCuda(  cudaSetDevice(0)  );
    
    checkCuda(  cudaMalloc((void**)&dev_c, numBytes)  );
    checkCuda(  cudaMalloc((void**)&dev_a, numBytes)  );
    checkCuda(  cudaMalloc((void**)&dev_b, numBytes)  );
    
    checkCuda(  cudaMemcpy(dev_a, a, numBytes, cudaMemcpyHostToDevice)  );
    checkCuda(  cudaMemcpy(dev_b, b, numBytes, cudaMemcpyHostToDevice)  );

    matrixMulTiled <<<grid, block>>> (dev_c, dev_a, dev_b);

    checkCuda(  cudaDeviceSynchronize()  );

    checkCuda(  cudaGetLastError()  );
    
    checkCuda(  cudaMemcpy(c, dev_c, numBytes, cudaMemcpyDeviceToHost) );

    cudaFree(   dev_c   );
    cudaFree(   dev_a   );
    cudaFree(   dev_b   );
}

int MatrixBank(float* c, const float* a, const float* b)
{
    int numBytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(MATRIX_SIZE / block.x, MATRIX_SIZE / block.y);

    checkCuda(cudaSetDevice(0));

    checkCuda(cudaMalloc((void**)&dev_c, numBytes));
    checkCuda(cudaMalloc((void**)&dev_a, numBytes));
    checkCuda(cudaMalloc((void**)&dev_b, numBytes));

    checkCuda(cudaMemcpy(dev_a, a, numBytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_b, b, numBytes, cudaMemcpyHostToDevice));

    matrixMultBank << <grid, block >> > (dev_c, dev_a, dev_b);

    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaGetLastError());

    checkCuda(cudaMemcpy(c, dev_c, numBytes, cudaMemcpyDeviceToHost));

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return 0;
}

void MatrixMulStream(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b)
{

    const int nStream = 4; // число CUDA-потоков
    int sizeMatrixStream = MATRIX_SIZE * MATRIX_SIZE / nStream;
    int numBytes = sizeMatrixStream * sizeof(float);

    BASE_TYPE* dev_a = 0;
    BASE_TYPE* dev_b = 0;
    BASE_TYPE* dev_c = 0;

    checkCuda(   cudaSetDevice(0));

    checkCuda(   cudaMalloc((void**)&dev_c, numBytes * nStream));
    checkCuda(   cudaMalloc((void**)&dev_a, numBytes * nStream));
    checkCuda(   cudaMalloc((void**)&dev_b, numBytes * nStream));

    cudaStream_t stream[nStream];

    for (size_t i = 0; i < nStream; ++i)
        cudaStreamCreate(&stream[i]);

    for (size_t i = 0; i < nStream; ++i)
    {
        cudaMemcpyAsync(dev_a + i * sizeMatrixStream, a + i * sizeMatrixStream, numBytes, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(dev_b + i * sizeMatrixStream, b + i * sizeMatrixStream, numBytes, cudaMemcpyHostToDevice, stream[i]);
    }

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(MATRIX_SIZE / block.x , MATRIX_SIZE / block.y / nStream);

    for (size_t i = 0; i < nStream; ++i) // Запуск функции-ядра
    {
        matrixDeviceStream << <grid, block, 0, stream[i] >> > (&dev_c[i * sizeMatrixStream], &dev_a[i * sizeMatrixStream], dev_b, MATRIX_SIZE / nStream);
    }

    checkCuda(   cudaDeviceSynchronize());

    checkCuda(   cudaGetLastError());

    for (size_t i = 0; i < nStream; ++i) 
        cudaMemcpyAsync(c + i * sizeMatrixStream, dev_c + i * sizeMatrixStream, numBytes, cudaMemcpyDeviceToHost, stream[i]);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    for (size_t i = 0; i < nStream; ++i)
        cudaStreamDestroy(stream[i]);
}

void MatrixMul(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b)
{
    int numBytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(BASE_TYPE);

    BASE_TYPE* dev_a = 0;
    BASE_TYPE* dev_b = 0;
    BASE_TYPE* dev_c = 0;



    checkCuda(   cudaSetDevice(0)   );

    checkCuda(   cudaMalloc((void**)&dev_c, numBytes)   );
    checkCuda(   cudaMalloc((void**)&dev_a, numBytes)   );
    checkCuda(   cudaMalloc((void**)&dev_b, numBytes)   );

    checkCuda(   cudaMemcpy(dev_a, a, numBytes, cudaMemcpyHostToDevice)   );
    checkCuda(   cudaMemcpy(dev_b, b, numBytes, cudaMemcpyHostToDevice)   );

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(MATRIX_SIZE /block.x, MATRIX_SIZE / block.y);
    matrixDevice << <grid, block >> > (dev_c , dev_a , dev_b );

    checkCuda(   cudaDeviceSynchronize());

    checkCuda(   cudaGetLastError());

    
    checkCuda(   cudaMemcpy(c, dev_c, numBytes, cudaMemcpyDeviceToHost)   );

    checkCuda(   cudaFree(dev_c)   );
    checkCuda(   cudaFree(dev_a)   );
    checkCuda(   cudaFree(dev_b)   );
}

void MatrixMuld(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b)
{


    int numBytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(BASE_TYPE);

    BASE_TYPE* dev_a = 0;
    BASE_TYPE* dev_b = 0;
    BASE_TYPE* dev_c = 0;



    checkCuda(cudaSetDevice(0));

    checkCuda(cudaMalloc((void**)&dev_c, numBytes));
    checkCuda(cudaMalloc((void**)&dev_a, numBytes));
    checkCuda(cudaMalloc((void**)&dev_b, numBytes));

    checkCuda(cudaMemcpy(dev_a, a, numBytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_b, b, numBytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(MATRIX_SIZE / block.x, MATRIX_SIZE / block.y);
    Muld << <grid, block >> > (dev_c, dev_a, dev_b);

    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaGetLastError());


    checkCuda(cudaMemcpy(c, dev_c, numBytes, cudaMemcpyDeviceToHost));

    checkCuda(cudaFree(dev_c));
    checkCuda(cudaFree(dev_a));
    checkCuda(cudaFree(dev_b));
}

void MatrixMulCublas(float* c, const float* a, const float* b)
{
    int numBytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);


   // cublasHandle_t handle;

    //cublasCreate(&handle);

    float* dev_a = 0;
    float* dev_b = 0;
    float* dev_c = 0;


    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(MATRIX_SIZE / block.x, MATRIX_SIZE / block.y);



    checkCuda(  cudaSetDevice(0));

    checkCuda(  cudaMalloc((void**)&dev_c, numBytes));
    checkCuda(  cudaMalloc((void**)&dev_a, numBytes));
    checkCuda(  cudaMalloc((void**)&dev_b, numBytes));

    checkCuda(  cudaMemcpy(dev_a, a, numBytes, cudaMemcpyHostToDevice));
    checkCuda(  cudaMemcpy(dev_b, b, numBytes, cudaMemcpyHostToDevice));

    float alpha = 1.0f;
    float beta = 0.0f;

    //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, &alpha, dev_a, MATRIX_SIZE, dev_b, MATRIX_SIZE, &beta, dev_c, MATRIX_SIZE);
    
    //cublasDestroy(handle);
    
    checkCuda(  cudaDeviceSynchronize());

    checkCuda(  cudaGetLastError());

    checkCuda(  cudaMemcpy(c, dev_c, numBytes, cudaMemcpyDeviceToHost));

    cudaFree(   dev_c   );
    cudaFree(   dev_a   );
    cudaFree(   dev_b   );
}

void matrixDeviceBuffB(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b)
{
    int numBytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(BASE_TYPE);

    BASE_TYPE* dev_a = 0;
    BASE_TYPE* dev_b = 0;
    BASE_TYPE* dev_c = 0;

    checkCuda(cudaSetDevice(0));

    checkCuda(cudaMalloc((void**)&dev_c, numBytes));
    checkCuda(cudaMalloc((void**)&dev_a, numBytes));
    checkCuda(cudaMalloc((void**)&dev_b, numBytes));

    checkCuda(cudaMemcpy(dev_a, a, numBytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_b, b, numBytes, cudaMemcpyHostToDevice));

    //dim3 grid(TS, TS);
    //dim3 block((MATRIX_SIZE + TS - 1) / TS, (MATRIX_SIZE + TS - 1) / TS);

    gpu_buf_t tB(MATRIX_SIZE * MATRIX_SIZE);
    dim3 gridT(TRX, TRY);
    dim3 blockT((MATRIX_SIZE + TRX - 1) / TRX, (MATRIX_SIZE + TRY - 1) / TRY);

    dim3 grid(TS, TS / WPT);
    dim3 block((MATRIX_SIZE + TS - 1) / TS, (MATRIX_SIZE + TS - 1) / TS);

    //matrixTest << <block, grid>> > (MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, dev_a, dev_b, dev_c);
    transposeBuffB << <blockT, gridT >> > (MATRIX_SIZE, MATRIX_SIZE, dev_b, tB.p);
    MatrixBuffB << <block, grid >> > (MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, dev_a, tB.p, dev_c);

    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaGetLastError());


    checkCuda(cudaMemcpy(c, dev_c, numBytes, cudaMemcpyDeviceToHost));

    checkCuda(cudaFree(dev_c));
    checkCuda(cudaFree(dev_a));
    checkCuda(cudaFree(dev_b));
}

void matrixDeviceBuffA(BASE_TYPE* c, const BASE_TYPE* a, const BASE_TYPE* b)
{
    int numBytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(BASE_TYPE);

    BASE_TYPE* dev_a = 0;
    BASE_TYPE* dev_b = 0;
    BASE_TYPE* dev_c = 0;

    checkCuda(cudaSetDevice(0));

    checkCuda(cudaMalloc((void**)&dev_c, numBytes));
    checkCuda(cudaMalloc((void**)&dev_a, numBytes));
    checkCuda(cudaMalloc((void**)&dev_b, numBytes));

    checkCuda(cudaMemcpy(dev_a, a, numBytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dev_b, b, numBytes, cudaMemcpyHostToDevice));


    gpu_buf_t tA(MATRIX_SIZE * MATRIX_SIZE);
    dim3 gridT(TRX, TRY);
    dim3 blockT((MATRIX_SIZE + TRX - 1) / TRX, (MATRIX_SIZE + TRY - 1) / TRY);

    dim3 grid(TSM / WPTM, TSN / WPTN);
    dim3 block(MATRIX_SIZE / TSM, MATRIX_SIZE / TSN);

    transpose << <blockT, gridT >> > (MATRIX_SIZE, MATRIX_SIZE, dev_a, tA.p);
    gemm << <block, grid >> > (MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE, tA.p, dev_b, dev_c);

    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaGetLastError());


    checkCuda(cudaMemcpy(c, dev_c, numBytes, cudaMemcpyDeviceToHost));

    checkCuda(cudaFree(dev_c));
    checkCuda(cudaFree(dev_a));
    checkCuda(cudaFree(dev_b));
}

//
//void df() {
//    // Size (in bytes) of matrix
//    size_t bytes = N * N * sizeof(int);
//
//    // Host vectors
//    vector<int> h_a(N * N);
//    vector<int> h_b(N * N);
//    vector<int> h_c(N * N);
//
//    // Initialize matrices
//    generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
//    generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });
//
//    // Allocate device memory
//    int* d_a, * d_b, * d_c;
//    cudaMalloc(&d_a, bytes);
//    cudaMalloc(&d_b, bytes);
//    cudaMalloc(&d_c, bytes);
//
//    // Copy data to the device
//    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
//
//    // Threads per CTA dimension
//    int THREADS = 32;
//
//    // Blocks per grid dimension (assumes THREADS divides N evenly)
//    int BLOCKS = N / THREADS;
//
//    // Use dim3 structs for block  and grid dimensions
//    dim3 threads(THREADS, THREADS);
//    dim3 blocks(BLOCKS, BLOCKS);
//
//    // Launch kernel
//    matrixMul << <blocks, threads >> > (d_a, d_b, d_c);
//
//    // Copy back to the host
//    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
//
//    // Check result
//    verify_result(h_a, h_b, h_c);
//
//    cout << "COMPLETED SUCCESSFULLY\n";
//
//    // Free memory on device
//    cudaFree(d_a);
//    cudaFree(d_b);
//    cudaFree(d_c);
//}