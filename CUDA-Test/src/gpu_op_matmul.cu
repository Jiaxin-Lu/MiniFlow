#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}
const int THREADS_PER_BLOCK = 512;
const int THREADS_PER_BLOCK_H = 256;
__global__ void array_set_kernel(float A, float val, int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
  {
    A[i] = val;
  }
}
int DLGpuArraySet(DLArrayHandle arr, float value) { 
  int n = 1;
  for (int i=0;i<arr->ndim;++i) n *= arr->shape[i];
  float* input_val = (float *)arr->data;
  int nblocks = (n+THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK;
  array_set_kernel<<<nblocks, THREADS_PER_BLOCK>>>(input_val, value, n);
  return 0;
}

__global__ void broadcast_to_kernel(const float* input, float *output, int in_n, int out_n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < out_n)
  {
    output[i] = input[i%in_n];
  }
}
int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  int in_n = 1;
  for (int i=0;i<input->ndim;++i) n *= input->shape[i];
  int out_n = 1;
  for (int i=0;i<output->ndim;++i) n *= output->shape[i];
  const float* input_val = (const float*)input->data;
  float* output_val = (float*)output->data;
  int nblocks = (n+THREADS_PER_BLOCK_H -1) / THREADS_PER_BLOCK_H;
  broadcast_to_kernel<<<nblocks, THREADS_PER_BLOCK_H>>>(input_val, output_val, in_n, out_n);

  return 0;
}
__global__ void reducesum_axis_zero_kernel(const float* input, float *output, int m, int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
  {
    output[i] = 0;
    for (int k=0;k<m;++k) output[i] += input[i+n*k];
  }
}
int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  int axisa = input->shape[0];
  int axisb = 1;
  for (int i=1;i<input->ndim;++i) axisb *= input->shape[i];
  const float* input_val = (const float*)input->data;
  float* output_val = (float*) output->data;
  int nblocks = (n+THREADS_PER_BLOCK_H -1) / THREADS_PER_BLOCK_H;
  reducesum_axis_zero_kernel<<<nblocks, THREADS_PER_BLOCK_H>>>(input_val, output_val, axisa, axisb);

  return 0;
}
__global__ void matrix_elementwise_add_kernel(const float *A, const float *B, float *output, int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
  {
    output[i] = A[i] + B[i];
  }
}
int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  int n = 1;
  for (int i=0;i<matA->ndim;++i) n *= matA->shape[i];
  const float* matA_ = (const float*)matA->data;
  const float* matB_ = (const float*)matB->data;
  float *output_val = (float*) output->data;
  int nblocks = (n+THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK;
  matrix_elementwise_add_kernel<<<nblocks, THREADS_PER_BLOCK>>>(matA_,matB_,output_val,n);
  return 0;
}
__global__ void matrix_elementwise_add_by_const_kernel(const float *A, const float val, float *output, int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<n)
  {
    output[i] = A[i] + val;
  }
}
int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  int n = 1;
  for (int i=0;i<input->ndim;++i) n *= input->shape[i];
  const float* matA_ = (const float*)input->data;
  float *output_val = (float*) output->data;
  int nblocks = (n+THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK;
  matrix_elementwise_add_by_const_kernel<<<nblocks, THREADS_PER_BLOCK>>>(matA_,val,output_val,n);
  return 0;
}

__global__  void matrix_elementwise_multiply_kernel(const float *A, const float *B, float *output, index_t n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
  {
    output[i] = A[i]*B[i];
  }
}
int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  int n = 1;
  for (int i=0;i<matA->ndim;++i) n*=matA->shape[i];
  const float* mata_ = (const float*) matA->data;
  const float* matb_ = (const float*) matB->data;
  float *output_val = (float *)output->data;
  int nblocks = (n+THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK;
  matrix_elementwise_multiply_kernel<<<nblocks, THREADS_PER_BLOCK>>>(mata_,matb_,output_val,n);
  return 0;
}

__global__ void matrix_elementwise_multiply_by_const_kernel(const float *A, const float val, float *output, int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<n)
  {
    output[i] = A[i] * val;
  }
}
int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  int n = 1;
  for (int i=0;i<input->ndim;++i) n *= input->shape[i];
  const float* matA_ = (const float*)input->data;
  float *output_val = (float*) output->data;
  int nblocks = (n+THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK;
  matrix_elementwise_multiply_by_const_kernel<<<nblocks, THREADS_PER_BLOCK>>>(matA_,val,output_val,n);
  return 0;
}
#define BLOCK_SIZE 16
// void matmul_COPY(const float* A, float* B, int &x, int &y, bool trans)
// {
//   B = (float*)malloc(x*y*sizeof(float));
//   if (!trans)
//   {
//     for (int xx=0;xx<x;++xx)
//       for (int yy=0;yy<y;++yy)
//       {
//         *(B+xx*y+yy) = *(A+xx*y+yy);
//       }
//   }
//   else
//   {
//     for (int xx=0;xx<x;++xx)
//       for (int yy=0;yy<y;++yy)
//       {
//         *(B+yy*x+xx) = *(A+xx*y+yy);
//       }
//     int t = x;
//     x = y, y = t;
//   }
// }
__global__ void mat_multiply_kernel(const float* l, const float* r, float* res, int dim, bool TA, bool TB)
{
  int i,j;
  float tmp = 0;
  __shared__ float Left_shared_t [BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Right_shared_t[BLOCK_SIZE][BLOCK_SIZE];
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  for (int t=0;t<grid_dim.x;++t)
  {
    j = t*BLOCK_SIZE + threadIdx.x;
    i = t*BLOCK_SIZE + threadIdx.y;
    int idl = TA?j*dim+row:row*dim+j;
    int idr = TB?col*dim+i:i*dim+col;
    Left_shared_t[threadIdx.y][threadIdx.x] = l[idl];
    Right_shared_t[threadIdx.y][threadIdx.x] = r[idr];
    __syncthreads();
    for (int k=0;k<BLOCK_SIZE;++k)
    {
      tmp += Left_shared_t[threadIdx.y][k] * Right_shared_t[k][threadIdx.x];
    }
    __syncthreads();
  }
  res[row*dim + col] = tmp;
}
inline void swap(int &x, int &y)
{
  int t = x;
  x = y, y = t;
}
int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  int lx = matA->shape[0], ly = matA->shape[1];
  int rx = matB->shape[0], ry = matB->shape[1];
  // float *lvh, *rvh;
  // const float* lvh_ = (const float*)matA->data;
  // const float* rvh_ = (const float*)matB->data;
  // matmul_COPY(lvh_, lvh, lx, ly, transposeA);
  // matmul_COPY(rvh_, rvh, rx, ry, transposeB);
  const float *lvh = (const float*)matA->data;
  const float *rvh = (const float*)matB->data;
  float* output_val = (float*)matC->data;
  if (transposeA) swap(lx, ly);
  if (transposeB) swap(rx, ry);
  int sqr_dim_x = max(lx,rx), sqr_dim_y = max(ly,ry);
  int size = max(sqr_dim_x, sqr_dim_y);
  int tmp = (size-1)/BLOCK_SIZE + 1;
  //int tmp = size / BLOCK_SIZE + (size % BLOCK_SIZE == 0 ? 0:1);
  size = tmp * BLOCK_SIZE;
  size_all = size * size * sizeof(float);
  // float *res_h = (float *)matC->data;
  // float *lvd, *rvd, *res_d;
  // cudaMalloc((void **) &lvd, size_all);
  // cudaMalloc((void **) &rvd, size_all);
  // cudaMalloc((void **) &res_d, size_all);

  // cudaMemcpy(lvd, lvh, size_all, cudaMemcpyHostToDevice);
  // cudaMemcpy(rvd, rvh, size_all, cudaMemcpyHostToDevice);

  dim3 Block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_dim(size/BLOCK_SIZE, size/BLOCK_SIZE);
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start,0);
  mat_multiply_kernel<<<grid_dim, Block_dim>>>(lvd, rvd, output_val, size, transposeA, transposeB);
  // cudaMemcpy(res_h, res_d, size_all, cudaMemcpyHostToDevice);
  // cudaEventRecord(stop,0);
  // cudaEventSynchronize(stop);
  // cudaEventDestroy(start);
  // cudaEventDestroy(stop);
  // free(lvh); free(rvh);
  //UNDONE! PUT RES_H BACK TO C!
  /* TODO: Your code here */
  // Hint: DO NOT use cublas
  return 0;
}
__global__ void relu_kernel(const float *A, float *output, int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
  {
    output[i] = (A[i] > 0) ? A[i]:0;
  }
}
int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  int n = 1;
  for (int i=0;i<input->ndim;++i) n*=input->shape[i];
  const float* input_val = (const float*)input->data;
  float* output_val = (float*)output->data;
  int nblocks = (n+THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK;
  relu_kernel<<<nblocks, THREADS_PER_BLOCK>>>(input_val, output_val, n)  
  return 0;
}

__global__ void relu_gradient_kernel(const float *A, const float *B, float *output, int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
  {
    output[i] = (A[i] > 0) ? B[i]:0;
  }
}
int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  int n = 1;
  for (int i=0;i<input->ndim;++i) n*=input->shape[i];
  const float* input_val = (const float*)input->data;
  const float* input_grad_val = (const float*)in_grad->data;
  float* output_val = (float*)output->data;
  int nblocks = (n+THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK;
  relu_gradient_kernel<<<nblocks, THREADS_PER_BLOCK>>>(input_val, input_grad_val, output_val, n);
  return 0;
}

__global__ softmax_kernel(const float* input, float* output, int nr, int nl)
{
  int i = blockDim.x * blockIdx.y * blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;
  if (i >= nr) return;
  input += i*nl;
  output += i*nl;
  float mx = *input;
  for (int x=1;x<nl;++x)
  {
    mx = max(mx, input[x]);
  }
  float sum = 0.0;
  for (int x=0;x<nl;++x)
  {
    sum += exp(input[x]-mx);
  }
  for (int x=0;x<nl;++x)
  {
    output[x] = exp(input[x]-mx)/sum;
  }
  
}
int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  int nr = input->shape[0];
  int nl = input->shape[1];
  const float* input_val = (const float*)input->data;
  float* output_val = (float*)output->data;
  dim3 threads;
  if (nr < 1024)
  {
    threads.x = nr;
  }
  else
  {
    threads.x = 1024;
    threads.y = (nr + 1023)/1024;
  }
  softmax_kernel<<<1, threads, nr*sizeof(float)>>>(input_val, output_val, nr, nl);
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
