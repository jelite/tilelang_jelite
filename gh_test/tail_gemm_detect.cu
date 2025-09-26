// matmul_tail_tensorcore_hostapi.cu
// H100/H200 (Hopper) 전용: Tensor Core (WMMA) 사용 + tail effect 분석 (호스트 API 활용)

#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda;

#define CK(call) do { \
  cudaError_t e=(call); \
  if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); \
  } \
} while(0)

#include <mma.h>
using namespace nvcuda;

__global__ void matmul_wmma_opt(half* A, half* B, float* C,
                                int M, int N, int K)
{
  const int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

  // 블록 구성: warp 단위 (32 threads)
  int warpId = (threadIdx.x >> 5) + threadIdx.y * (blockDim.x >> 5);
  int warpsPerBlock = (blockDim.x * blockDim.y) >> 5;

  // warp마다 하나의 타일 담당
  int tileRow = (blockIdx.y * warpsPerBlock + warpId) * WMMA_M;
  int tileCol = blockIdx.x * WMMA_N;

  if (tileRow >= M || tileCol >= N) return;

  // Accumulator 초기화
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  // K dimension 루프
  for (int k0 = 0; k0 < K; k0 += WMMA_K) {
    if (k0 + WMMA_K <= K) {
      // load A (row major), B (col major)
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

      const half* tileA = A + tileRow * K + k0;
      const half* tileB = B + k0 * N + tileCol;

      wmma::load_matrix_sync(a_frag, tileA, K);
      wmma::load_matrix_sync(b_frag, tileB, N);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }

  // 결과 저장
  wmma::store_matrix_sync(C + tileRow * N + tileCol, c_frag, N, wmma::mem_row_major);
}


// WMMA 기반 Tensor Core matmul (FP16 input, FP32 accumulate)
__global__ void matmul_wmma(half* A, half* B, float* C,
                            int M, int N, int K)
{
  const int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

  // 단일 워프만 사용 (데모 목적)
  const int warp_id = (threadIdx.y >> 5);
  if (warp_id != 0) return;

  const int row = blockIdx.y * WMMA_M;
  const int col = blockIdx.x * WMMA_N;
  if (row >= M || col >= N) return;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < K; k0 += WMMA_K) {
    if ((k0 + WMMA_K) <= K) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

      const half* tileA = A + row * K + k0;
      const half* tileB = B + k0 * N + col;

      wmma::load_matrix_sync(a_frag, tileA, K);
      wmma::load_matrix_sync(b_frag, tileB, N);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }

  wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
}

// ---------------- Host -----------------
int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s M N K\n", argv[0]);
    return 1;
  }
  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int K = atoi(argv[3]);

  // 장치 속성
  cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
  const int num_sms = prop.multiProcessorCount;

  // 블록/그리드 설정 (WMMA 16x16 타일당 블록 1개)
  dim3 block(32, 32, 1); // 단일 warp만 사용 (데모 목적)
  dim3 grid((N + 16 - 1) / 16, (M + 16 - 1) / 16, 1);
  const int total_blocks = grid.x * grid.y * grid.z;

  // 호스트 API로 occupancy 계산
  int maxBlocksPerSM = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocksPerSM,
      matmul_wmma,
      block.x * block.y * block.z,
      0));
  if (maxBlocksPerSM <= 0) maxBlocksPerSM = 1;

  const int wave = num_sms * maxBlocksPerSM;
  const int full_wave_blocks = (wave > 0) ? (total_blocks / wave) * wave : total_blocks;
  const int tail_blocks = total_blocks - full_wave_blocks;
  const int tail_start_block = (tail_blocks > 0) ? full_wave_blocks : -1;

  // 메모리 할당
  size_t sizeA = (size_t)M * K * sizeof(half);
  size_t sizeB = (size_t)K * N * sizeof(half);
  size_t sizeC = (size_t)M * N * sizeof(float);

  half *A, *B; float *C;
  CK(cudaMallocManaged(&A, sizeA));
  CK(cudaMallocManaged(&B, sizeB));
  CK(cudaMallocManaged(&C, sizeC));

  for (size_t i = 0; i < (size_t)M*K; i++) A[i] = __float2half(1.0f);
  for (size_t i = 0; i < (size_t)K*N; i++) B[i] = __float2half(1.0f);

  // 실행
  matmul_wmma<<<grid, block>>>(A, B, C, M, N, K);
  CK(cudaDeviceSynchronize());

  // 결과 출력
  printf("[Info] GPU=%s, SMs=%d\n", prop.name, num_sms);
  printf("[Info] M=%d N=%d K=%d\n", M, N, K);
  printf("[Info] grid=(%d,%d,%d) block=(%d,%d,%d)\n",
         grid.x, grid.y, grid.z, block.x, block.y, block.z);
  printf("[Info] total_blocks=%d, wave=%d (num_sms*maxBlocksPerSM=%d*%d)\n",
         total_blocks, wave, num_sms, maxBlocksPerSM);
  printf("[Info] full_wave_blocks=%d, tail_blocks=%d\n",
         full_wave_blocks, tail_blocks);

  if (tail_start_block >= 0) {
    printf("[Result] Tail wave 시작 블록 = %d\n", tail_start_block);
  } else {
    printf("[Result] Tail wave 없음\n");
  }

  cudaFree(A); cudaFree(B); cudaFree(C);
  return 0;
}
