// matmul_wmma_cublas_compare.cu
// baseline kernel vs optimized WMMA kernel vs cuBLAS GEMM
#include <cuda_runtime.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace nvcuda;
#define CK(call) do { \
  cudaError_t e=(call); \
  if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); \
  } \
} while(0)

static inline int iDivUp(int a, int b){ return (a + b - 1) / b; }

// ---------------- Naive kernel ----------------
__global__ void matmul_naive(const half* A, const half* B, float* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    acc += __half2float(A[row*K + k]) * __half2float(B[k*N + col]);
  }
  C[row*N + col] = acc;
}

// ---------------- Optimized WMMA kernel ----------------
__global__ void matmul_wmma_opt(const half* A, const half* B, float* C, int M, int N, int K) {
  constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
  int warpsX = blockDim.x >> 5;
  int warpsY = blockDim.y;
  int warpsPerBlock = warpsX * warpsY;
  int warpIdxInBlock = (threadIdx.x >> 5) + threadIdx.y * warpsX;
  int lane = threadIdx.x & 31;
  int tileY = blockIdx.y * warpsPerBlock + warpIdxInBlock;
  int tileX = blockIdx.x;
  int row0 = tileY * WMMA_M;
  int col0 = tileX * WMMA_N;
  if (row0 >= M || col0 >= N) return;

  extern __shared__ unsigned char s[];
  int bytesA = WMMA_M*WMMA_K*sizeof(half);
  int bytesB = WMMA_K*WMMA_N*sizeof(half);
  int bytesC = WMMA_M*WMMA_N*sizeof(float);
  int perWarpBytes = bytesA + bytesB + bytesC;
  unsigned char* warpBase = s + warpIdxInBlock*perWarpBytes;
  half* smemA = reinterpret_cast<half*>(warpBase);
  half* smemB = reinterpret_cast<half*>(warpBase + bytesA);
  float* smemC = reinterpret_cast<float*>(warpBase + bytesA + bytesB);

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < K; k0 += WMMA_K) {
    for (int i = lane; i < WMMA_M*WMMA_K; i+=32) {
      int r=i/WMMA_K, c=i%WMMA_K;
      int gRow=row0+r, gCol=k0+c;
      smemA[i] = (gRow<M && gCol<K)? A[gRow*K+gCol]:__float2half(0.f);
    }
    for (int i = lane; i < WMMA_K*WMMA_N; i+=32) {
      int r=i/WMMA_N, c=i%WMMA_N;
      int gRow=k0+r, gCol=col0+c;
      smemB[i] = (gRow<K && gCol<N)? B[gRow*N+gCol]:__float2half(0.f);
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, WMMA_M,WMMA_N,WMMA_K,half,wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M,WMMA_N,WMMA_K,half,wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, smemA, WMMA_K);
    wmma::load_matrix_sync(b_frag, smemB, WMMA_N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncwarp();
  }

  wmma::store_matrix_sync(smemC, c_frag, WMMA_N, wmma::mem_row_major);
  __syncwarp();
  for (int i=lane; i<WMMA_M*WMMA_N; i+=32) {
    int r=i/WMMA_N, c=i%WMMA_N;
    int gRow=row0+r, gCol=col0+c;
    if (gRow<M && gCol<N) C[gRow*N+gCol] = smemC[r*WMMA_N+c];
  }
}

// ---------------- Main ----------------
int main(int argc, char** argv){
  if (argc!=4){ printf("Usage: %s M N K\n",argv[0]); return 1;}
  int M=atoi(argv[1]), N=atoi(argv[2]), K=atoi(argv[3]);
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
  printf("[Info] GPU=%s\n",prop.name);

  size_t sizeA=(size_t)M*K, sizeB=(size_t)K*N, sizeC=(size_t)M*N;
  half *A,*B; float *C_ref,*C_opt,*C_blas;
  CK(cudaMallocManaged(&A,sizeA*sizeof(half)));
  CK(cudaMallocManaged(&B,sizeB*sizeof(half)));
  CK(cudaMallocManaged(&C_ref,sizeC*sizeof(float)));
  CK(cudaMallocManaged(&C_opt,sizeC*sizeof(float)));
  CK(cudaMallocManaged(&C_blas,sizeC*sizeof(float)));
  for(size_t i=0;i<sizeA;i++) A[i]=__float2half(1.f);
  for(size_t i=0;i<sizeB;i++) B[i]=__float2half(1.f);

  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);

  // baseline
  dim3 blk(32,8); dim3 grd(iDivUp(N,blk.x),iDivUp(M,blk.y));
  cudaEventRecord(s);
  matmul_naive<<<grd,blk>>>(A,B,C_ref,M,N,K);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms_ref; cudaEventElapsedTime(&ms_ref,s,e);
  printf("[Timing] naive=%.3f ms\n",ms_ref);

  // opt
  dim3 blkOpt(128,4); int warps=(blkOpt.x>>5)*blkOpt.y;
  dim3 grdOpt(iDivUp(N,16), iDivUp(iDivUp(M,16),warps));
  size_t shmem=warps*(16*16*sizeof(half)*2+16*16*sizeof(float));
  cudaEventRecord(s);
  matmul_wmma_opt<<<grdOpt,blkOpt,shmem>>>(A,B,C_opt,M,N,K);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms_opt; cudaEventElapsedTime(&ms_opt,s,e);
  printf("[Timing] opt WMMA=%.3f ms\n",ms_opt);

  // cuBLAS
  cublasHandle_t h; cublasCreate(&h);
  float alpha=1.f,beta=0.f;
  cudaEventRecord(s);
  cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,
               N,M,K,
               &alpha,
               B,CUDA_R_16F,N,
               A,CUDA_R_16F,K,
               &beta,
               C_blas,CUDA_R_32F,N,
               CUDA_R_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms_blas; cudaEventElapsedTime(&ms_blas,s,e);
  printf("[Timing] cuBLAS=%.3f ms\n",ms_blas);
  cublasDestroy(h);

  // check
  int errs=0;
  for(size_t i=0;i<sizeC;i++){
    if (fabsf(C_ref[i]-C_opt[i])>1e-2f || fabsf(C_ref[i]-C_blas[i])>1e-2f){
      if(errs<5) printf("Mismatch at %zu: ref=%f opt=%f blas=%f\n",i,C_ref[i],C_opt[i],C_blas[i]);
      errs++;
    }
  }
  if(errs==0) printf("[Check] All match ✅\n");
  else printf("[Check] Errors=%d ❌\n",errs);

  cudaFree(A);cudaFree(B);cudaFree(C_ref);cudaFree(C_opt);cudaFree(C_blas);
  return 0;
}
