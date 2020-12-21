#pragma once
void gemm_v1(int M, int N, int K, const float* A, const float* B, float* C);
void gemm_v2(int M, int N, int K, const float* A, const float* B, float* C);
void micro_6x16(int K, const float* A, int lda, int step, const float* B, int ldb, float* C, int ldc);
void micro_4x16(int K, const float* A, int lda, int step,
    const float* B, int ldb, float* C, int ldc);
void micro_2x16(int K, const float* A, int lda, int step,
    const float* B, int ldb, float* C, int ldc);
void init_c(int M, int N, float* C, int ldc);
void gemm_v3(int M, int N, int K, const float* A, const float* B, float* C);
void reorder_b_16(int K, const float* B, int ldb, float* bufB);
void gemm_v4(int M, int N, int K, const float* A, const float* B, float* C);
void macro_v5(int M, int N, int K, const float* A, int lda,
    const float* B, int ldb, float* bufB, float* C, int ldc);
void gemm_v5(int M, int N, int K, const float* A, const float* B, float* C);
void reorder_a_6(const float* A, int lda, int M, int K, float* bufA);
void macro_v6(int M, int N, int K, const float* A,
    const float* B, int ldb, float* bufB, float* C, int ldc);
void gemm_v6(int M, int N, int K, const float* A, const float* B, float* C);
void macro_v7(int M, int N, int K, const float* A,
    const float* B, int ldb, float* bufB, float* C, int ldc);
void gemm_v7(int M, int N, int K, const float* A, const float* B, float* C);

