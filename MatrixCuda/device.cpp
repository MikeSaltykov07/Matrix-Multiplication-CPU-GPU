#include <iostream>
#include < stdio.h  >
#include < cstdlib   >
#include "immintrin.h"

#pragma optimize("", off)
void release_mode_breakpoint()
{
    int put_breakpoint_here = 1;
}
#pragma optimize("", on)

void gemm_v1(int M, int N, int K, const float* A, const float* B, float* C)
{
    for (int i = 0; i < M; ++i)
    {
        float* c = C + i * N;
        for (int j = 0; j < N; ++j)
            c[j] = 0;
        for (int k = 0; k < K; ++k)
        {
            const float* b = B + k * N;
            float a = A[i * K + k];
            for (int j = 0; j < N; ++j)
                c[j] += a * b[j];
        }
    }
}

void gemm_v2(int M, int N, int K, const float* A, const float* B, float* C)
{
    for (int i = 0; i < M; ++i)
    {
        float* c = C + i * N;
        for (int j = 0; j < N; j += 8)
            _mm256_storeu_ps(c + j + 0, _mm256_setzero_ps());

        for (int k = 0; k < K; ++k)
        {
            const float* b = B + k * N;
            __m256 a = _mm256_set1_ps(A[i * K + k]);
            for (int j = 0; j < N; j += 16)
            {
                _mm256_storeu_ps(c + j + 0, _mm256_fmadd_ps(a, _mm256_loadu_ps(b + j + 0), _mm256_loadu_ps(c + j + 0)));
                _mm256_storeu_ps(c + j + 8, _mm256_fmadd_ps(a, _mm256_loadu_ps(b + j + 8), _mm256_loadu_ps(c + j + 8)));
            }
        }
    }
}


void micro_6x16(int K, const float* A, int lda, int step, const float* B, int ldb, float* C, int ldc)
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

void micro_4x16(int K, const float* A, int lda, int step, const float* B, int ldb, float* C, int ldc)
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
    __m256 b0, b1, a0, a1;
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
}

void micro_2x16(int K, const float* A, int lda, int step, const float* B, int ldb, float* C, int ldc)
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
    __m256 b0, b1, a0, a1;
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
        B += ldb; A += step;
    }
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c00, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c01, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c10, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c11, _mm256_loadu_ps(C + 8)));
}

void init_c(int M, int N, float* C, int ldc)
{
    for (int i = 0; i < M; ++i, C += ldc)
        for (int j = 0; j < N; j += 8)
            _mm256_storeu_ps(C + j, _mm256_setzero_ps());
}

void gemm_v3(int M, int N, int K, const float* A, const float* B, float* C)
{
    for (int i = 0; i < M; i += 6)
    {
        for (int j = 0; j < N; j += 16)
        {
            init_c(6, 16, C + i * N + j, N);
            micro_6x16(K, A + i * K, K, 1, B + j, N, C + i * N + j, N);
        }
    }
}

struct buf_t
{
    float* p;
    int n;

    buf_t(int size) : n(size), p((float*)_mm_malloc(size * 4, 64)) {}

    ~buf_t() { _mm_free(p); }
};

void reorder_b_16(int K, const float* B, int ldb, float* bufB)
{
    for (int k = 0; k < K; ++k, B += ldb, bufB += 16)
    {
        _mm256_storeu_ps(bufB + 0, _mm256_loadu_ps(B + 0));
        _mm256_storeu_ps(bufB + 8, _mm256_loadu_ps(B + 8));
    }
}

void gemm_v4(int M, int N, int K, const float* A, const float* B, float* C)
{
    for (int j = 0; j < N; j += 16)
    {
        buf_t bufB(16 * K);
        reorder_b_16(K, B + j, N, bufB.p);
        for (int i = 0; i < M; i += 6)
        {
            init_c(6, 16, C + i * N + j, N);
            micro_6x16(K, A + i * K, K, 1, bufB.p, 16, C + i * N + j, N);
        }
    }
}


void macro_v5(int M, int N, int K, const float* A, int lda,
    const float* B, int ldb, float* bufB, float* C, int ldc)
{
    for (int j = 0; j < N; j += 16)
    {
        reorder_b_16(K, B + j, ldb, bufB);
        for (int i = 0; i < M; i += 6)
            micro_6x16(K, A + i * lda, lda, 1, bufB, 16, C + i * ldc + j, ldc);
    }
}


void gemm_v5(int M, int N, int K, const float* A, const float* B, float* C)
{
    const int L1 = 384 * 1024;
    int mK = std::min(L1 / 4 / 16, K);
    buf_t bufB(16 * mK);
    for (int k = 0; k < K; k += mK)
    {
        int dK = std::min(K, k + mK) - k;
        if (k == 0)
            init_c(M, N, C, N);
        macro_v5(M, N, dK, A + k, K, B + k * N, N, bufB.p, C, N);
    }
}

void reorder_a_6(const float* A, int lda, int M, int K, float* bufA)
{
    for (int i = 0; i < M; i += 6)
    {
        for (int k = 0; k < K; k += 4)
        {
            const float* pA = A + k;
            __m128 a0 = _mm_loadu_ps(pA + 0 * lda);
            __m128 a1 = _mm_loadu_ps(pA + 1 * lda);
            __m128 a2 = _mm_loadu_ps(pA + 2 * lda);
            __m128 a3 = _mm_loadu_ps(pA + 3 * lda);
            __m128 a4 = _mm_loadu_ps(pA + 4 * lda);
            __m128 a5 = _mm_loadu_ps(pA + 5 * lda);
            __m128 a00 = _mm_unpacklo_ps(a0, a2);
            __m128 a01 = _mm_unpacklo_ps(a1, a3);
            __m128 a10 = _mm_unpackhi_ps(a0, a2);
            __m128 a11 = _mm_unpackhi_ps(a1, a3);
            __m128 a20 = _mm_unpacklo_ps(a4, a5);
            __m128 a21 = _mm_unpackhi_ps(a4, a5);
            _mm_storeu_ps(bufA + 0, _mm_unpacklo_ps(a00, a01));
            _mm_storel_pi((__m64*)(bufA + 4), a20);
            _mm_storeu_ps(bufA + 6, _mm_unpackhi_ps(a00, a01));
            _mm_storeh_pi((__m64*)(bufA + 10), a20);
            _mm_storeu_ps(bufA + 12, _mm_unpacklo_ps(a10, a11));
            _mm_storel_pi((__m64*)(bufA + 16), a21);
            _mm_storeu_ps(bufA + 18, _mm_unpackhi_ps(a10, a11));
            _mm_storeh_pi((__m64*)(bufA + 22), a21);
            bufA += 24;
        }
        A += 6 * lda;
    }
}


void macro_v6(int M, int N, int K, const float* A,
    const float* B, int ldb, float* bufB, float* C, int ldc)
{
    for (int j = 0; j < N; j += 16)
    {
        reorder_b_16(K, B + j, ldb, bufB);
        for (int i = 0; i < M; i += 6)
            micro_6x16(K, A + i * K, 1, 6, bufB, 16, C + i * ldc + j, ldc);
    }
}

void gemm_v6(int M, int N, int K, const float* A, const float* B, float* C)
{
    const int L1 = 384 * 1024, L2 = 1536 * 1024;
    int mK = std::min(L1 / 4 / 16, K) / 4 * 4;
    int mM = std::min(L2 / 4 / mK, M) / 6 * 6;
    buf_t bufB(16 * mK);
    buf_t bufA(mK * mM);
    for (int k = 0; k < K; k += mK)
    {
        int dK = std::min(K, k + mK) - k;
        for (int i = 0; i < M; i += mM)
        {
            int dM = std::min(M, i + mM) - i;
            if (k == 0)
                init_c(dM, N, C + i * N, N);
            reorder_a_6(A + i * K + k, K, dM, dK, bufA.p);
            macro_v6(dM, N, dK, bufA.p, B + k * N, N, bufB.p, C + i * N, N);
        }
    }
}

void macro_v7(int M, int N, int K, const float* A,
    const float* B, int ldb, float* bufB, bool reorderB, float* C, int ldc)
{
    for (int j = 0; j < N; j += 16)
    {
        if (reorderB)
            reorder_b_16(K, B + j, ldb, bufB + K * j);
        for (int i = 0; i < M; i += 6)
            micro_6x16(K, A + i * K, 1, 6, bufB + K * j, 16, C + i * ldc + j, ldc);
    }
}

void gemm_v7(int M, int N, int K, const float* A, const float* B, float* C)
{
    const int L1 = 384 * 1024, L2 = 1536 * 1024, L3 = 12 * 1024 * 1024;
    int mK = std::min(L1 / 4 / 16, K) / 4 * 4;
    int mM = std::min(L2 / 4 / mK, M) / 6 * 6;
    int mN = std::min(L3 / 4 / mK, N) / 16 * 16;
    buf_t bufB(mN * mK);
    buf_t bufA(mK * mM);
    for (int j = 0; j < N; j += mN)
    {
        int dN = std::min(N, j + mN) - j;
        for (int k = 0; k < K; k += mK)
        {
            int dK = std::min(K, k + mK) - k;
            for (int i = 0; i < M; i += mM)
            {
                int dM = std::min(M, i + mM) - i;
                if (k == 0)
                    init_c(dM, dN, C + i * N + j, N);
                reorder_a_6(A + i * K + k, K, dM, dK, bufA.p);
                macro_v7(dM, dN, dK, bufA.p, B + k * N + j, N,
                    bufB.p, i == 0, C + i * N + j, N);
            }
        }
    }
}
