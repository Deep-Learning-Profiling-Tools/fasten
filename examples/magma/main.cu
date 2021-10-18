#include <iostream>
#include <random>
#include <cuda.h>
#include <cublas.h>
#include <magma_v2.h>

#define RUNTIME_API_CALL(apiFuncCall)                                        \
  do {                                                                       \
    cudaError_t _status = apiFuncCall;                                       \
    if (_status != cudaSuccess) {                                            \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
        __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));      \
      exit(-1);                                                              \
    }                                                                        \
  } while (0)

static magma_int_t dev = 0;
static magma_queue_t queue = NULL;
const static int M = 16384;
const static int K = 16;
const static int N = 8;
const static int B = 128;
static float *l_gpu = NULL;
static float *r_gpu = NULL;
static float *p_gpu = NULL;
static float *l_cpu = NULL;
static float *r_cpu = NULL;
static float *p_cpu = NULL;

static magma_int_t m_shapes_cpu[B + 1];
static magma_int_t n_shapes_cpu[B + 1];
static magma_int_t k_shapes_cpu[B + 1];
static magma_int_t *m_shapes_gpu;
static magma_int_t *n_shapes_gpu;
static magma_int_t *k_shapes_gpu;

static magma_int_t l_ldd_cpu[B + 1];
static magma_int_t r_ldd_cpu[B + 1];
static magma_int_t p_ldd_cpu[B + 1];
static magma_int_t *l_ldd_gpu;
static magma_int_t *r_ldd_gpu;
static magma_int_t *p_ldd_gpu;

static float *l_gpu_arr_cpu[B];
static float *r_gpu_arr_cpu[B];
static float *p_gpu_arr_cpu[B];
static float **l_gpu_arr_gpu;
static float **r_gpu_arr_gpu;
static float **p_gpu_arr_gpu;

/*
 *  [16384, 16] x [128, 16, 8] = [16384, 8]

 *  [16384 / 128, 16] x [16, 8]
 *  [16384 / 128, 16] x [16, 8]
 */
static void init_array(float *arr, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    arr[i] = static_cast<float>(i) / size;
  }
}

static void setup() {
  magma_init();
  magma_queue_create(dev, &queue);

  l_cpu = new float[M * K];
  r_cpu = new float[K * N * B];
  p_cpu = new float[M * N];

  RUNTIME_API_CALL(cudaMalloc(&l_gpu, sizeof(float) * M * K));
  RUNTIME_API_CALL(cudaMalloc(&r_gpu, sizeof(float) * K * N * B));
  RUNTIME_API_CALL(cudaMalloc(&p_gpu, sizeof(float) * M * N));

	RUNTIME_API_CALL(cudaMalloc(&m_shapes_gpu, sizeof(magma_int_t) * (B + 1)));
	RUNTIME_API_CALL(cudaMalloc(&n_shapes_gpu, sizeof(magma_int_t) * (B + 1)));
	RUNTIME_API_CALL(cudaMalloc(&k_shapes_gpu, sizeof(magma_int_t) * (B + 1)));

  RUNTIME_API_CALL(cudaMalloc(&l_ldd_gpu, sizeof(magma_int_t) * (B + 1)));
  RUNTIME_API_CALL(cudaMalloc(&r_ldd_gpu, sizeof(magma_int_t) * (B + 1)));
  RUNTIME_API_CALL(cudaMalloc(&p_ldd_gpu, sizeof(magma_int_t) * (B + 1)));

	RUNTIME_API_CALL(cudaMalloc(&l_gpu_arr_gpu, sizeof(float *) * B));
	RUNTIME_API_CALL(cudaMalloc(&r_gpu_arr_gpu, sizeof(float *) * B));
	RUNTIME_API_CALL(cudaMalloc(&p_gpu_arr_gpu, sizeof(float *) * B));
}

static void cleanup() {
  delete [] l_cpu;
  delete [] r_cpu;
  delete [] p_cpu;

  RUNTIME_API_CALL(cudaFree(l_gpu));
  RUNTIME_API_CALL(cudaFree(r_gpu));
  RUNTIME_API_CALL(cudaFree(p_gpu));

	RUNTIME_API_CALL(cudaFree(m_shapes_gpu));
	RUNTIME_API_CALL(cudaFree(n_shapes_gpu));
	RUNTIME_API_CALL(cudaFree(k_shapes_gpu));

	RUNTIME_API_CALL(cudaFree(l_ldd_gpu));
	RUNTIME_API_CALL(cudaFree(r_ldd_gpu));
	RUNTIME_API_CALL(cudaFree(p_ldd_gpu));

	RUNTIME_API_CALL(cudaFree(l_gpu_arr_gpu));
	RUNTIME_API_CALL(cudaFree(r_gpu_arr_gpu));
	RUNTIME_API_CALL(cudaFree(p_gpu_arr_gpu));

  magma_queue_destroy(queue);
  magma_finalize();
}

static void init() {
  init_array(l_cpu, M * K);
  init_array(r_cpu, K * N * B);
  RUNTIME_API_CALL(cudaMemcpy(l_gpu, l_cpu, M * K, cudaMemcpyHostToDevice));
  RUNTIME_API_CALL(cudaMemcpy(r_gpu, r_cpu, K * N * B, cudaMemcpyHostToDevice));

  for (size_t i = 0; i < B; ++i) {
    m_shapes_cpu[i] = M / B;
    n_shapes_cpu[i] = N;
    k_shapes_cpu[i] = K;
		// column major
    l_ldd_cpu[i] = m_shapes_cpu[i];
    r_ldd_cpu[i] = K;
    p_ldd_cpu[i] = m_shapes_cpu[i];
    l_gpu_arr_cpu[i] = l_gpu + i * m_shapes_cpu[i] * K;
    r_gpu_arr_cpu[i] = r_gpu + i * K * N;
    p_gpu_arr_cpu[i] = p_gpu + i * m_shapes_cpu[i] * N;
  }

	RUNTIME_API_CALL(cudaMemcpy(m_shapes_gpu, m_shapes_cpu, sizeof(magma_int_t) * B, cudaMemcpyHostToDevice));
	RUNTIME_API_CALL(cudaMemcpy(n_shapes_gpu, n_shapes_cpu, sizeof(magma_int_t) * B, cudaMemcpyHostToDevice));
	RUNTIME_API_CALL(cudaMemcpy(k_shapes_gpu, k_shapes_cpu, sizeof(magma_int_t) * B, cudaMemcpyHostToDevice));

  RUNTIME_API_CALL(cudaMemcpy(l_ldd_gpu, l_ldd_cpu, sizeof(magma_int_t) * B, cudaMemcpyHostToDevice));
  RUNTIME_API_CALL(cudaMemcpy(r_ldd_gpu, r_ldd_cpu, sizeof(magma_int_t) * B, cudaMemcpyHostToDevice));
  RUNTIME_API_CALL(cudaMemcpy(p_ldd_gpu, p_ldd_cpu, sizeof(magma_int_t) * B, cudaMemcpyHostToDevice));

	RUNTIME_API_CALL(cudaMemcpy(l_gpu_arr_gpu, l_gpu_arr_cpu, sizeof(float *) * B, cudaMemcpyHostToDevice));
	RUNTIME_API_CALL(cudaMemcpy(r_gpu_arr_gpu, r_gpu_arr_cpu, sizeof(float *) * B, cudaMemcpyHostToDevice));
	RUNTIME_API_CALL(cudaMemcpy(p_gpu_arr_gpu, p_gpu_arr_cpu, sizeof(float *) * B, cudaMemcpyHostToDevice));
}

static void compute() {
  cudaEvent_t start_event, end_event;
  RUNTIME_API_CALL(cudaEventCreate(&start_event));
  RUNTIME_API_CALL(cudaEventCreate(&end_event));

  RUNTIME_API_CALL(cudaEventRecord(start_event));
  magmablas_sgemm_vbatched(
    MagmaNoTrans,
    MagmaNoTrans,
    m_shapes_gpu,
    n_shapes_gpu,
    k_shapes_gpu,
    1.0,
    l_gpu_arr_gpu,
    l_ldd_gpu,
    r_gpu_arr_gpu,
    r_ldd_gpu,
    0.0,
    p_gpu_arr_gpu,
    p_ldd_gpu,
    B,
    queue);
  RUNTIME_API_CALL(cudaEventRecord(end_event));
  RUNTIME_API_CALL(cudaEventSynchronize(end_event));

  float ms = 0.0;
  RUNTIME_API_CALL(cudaEventElapsedTime(&ms, start_event, end_event));
	std::cout << "Elapsed time: " << ms << "ms" << std::endl;
  
  RUNTIME_API_CALL(cudaEventDestroy(start_event));
  RUNTIME_API_CALL(cudaEventDestroy(end_event));
}

int main() {
  setup();
  init();
  compute();
  cleanup();
  return 0;
}