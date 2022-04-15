/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/utsname.h>
#endif

// Includes, system
#include <math.h>
// #include <npp.h>
// #include <nppi.h>
#include <stdio.h>

#include <cassert>
// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>  // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>  // helper functions for CUDA error check
#include <pthread.h>
#include <semaphore.h>

#include <cstdint>

#include "cuda_image.h"
#include <vector>

#define LOGCUDA(...) printf("CUDA\t" __VA_ARGS__)
#define MAX_STREAM_COUNT 1

typedef struct cuda_dev_ctrl {
  /*decide which debug information you want present. refer to
   * cuda_debug-info-value defined above. */
  int32_t debug_info;
  pthread_mutex_t mutex;
  // gpu total number
  uint32_t device_count;
  // stream count
  uint32_t stream_count;
} cuda_dev_ctrl_t;


static uint32_t is_init = 0;
static cuda_dev_ctrl_t g_cuda_dev_ctrl;
static cudaStream_t cuda_stream_array[MAX_STREAM_COUNT];
///////////////////////////////////////////global
///kernel//////////////////////////////////////////////
__global__ void dev_yuv422_single_mask(size_t width, size_t height, uint8_t *yuv_in_out,
    size_t *mat, size_t num_mat) {

  size_t scalex, scaley, fx0, fy0;
  size_t i, j; //j=width, i=hight
  size_t threadid;
  size_t blockid;
  size_t *p;

  blockid = blockIdx.x + blockIdx.y * gridDim.x;
  threadid = blockid * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) +
             threadIdx.x;

  j = (threadid * 2) % width;
  i = ((threadid * 2) - j) / width;


  for(size_t l = 0; l < num_mat; l++){
    p = (size_t*)(mat + l * 4);
    fx0 = *p;
    fy0 = *(p + 1);
    scalex = *(p + 2);
    scaley = *(p + 3);
    if((j > fx0) 
        && (j < fx0 + scalex)
        && (i > fy0)
        && (i < fy0 + scaley)
      ){
      yuv_in_out[threadid * 4] = 0; //erase Y
      yuv_in_out[(threadid * 4) + 2] = 0; //erase pixel 2 ->Y
    }
  }
}

__global__ void dev_yuvn12_single_mask(size_t width, size_t height, uint8_t *yuyv_in_out, 
    size_t *mat, size_t num_mat) {

  size_t i, j;
  j = blockIdx.x * blockDim.x + threadIdx.x;
  i = blockIdx.y * blockDim.y + threadIdx.y;

  size_t scalex, scaley, fx0, fy0;
  size_t *p;
  for(size_t l = 0; l < num_mat; l++){
    p = (size_t*)(mat + l * 4);
    fx0 = *p;
    fy0 = *(p + 1);
    scalex = *(p + 2);
    scaley = *(p + 3);
    if((j > fx0) 
        && (j < fx0 + scalex)
        && (i > fy0)
        && (i < fy0 + scaley)
      ){
      yuyv_in_out[i * width + j] = 0; //y = black
    }
  }
}

///////////////////////////////////////////host
///funcs//////////////////////////////////////////////

static int host_yuv422_single_mask(uint8_t *yuvbuffer,
                              size_t width, size_t height, cuda_mat_t *mat, size_t num_mat,
                              cudaStream_t stream) {
  size_t iw, ih;
  if (NULL == yuvbuffer) {
    LOGCUDA("%s invalid param\n", __func__);
    return -1;
  }
  if(num_mat > 100){
    LOGCUDA("illegal mat num %zu", num_mat);
    return -1;
  }
  size_t *gpu_mat = (size_t*)cuda_malloc(num_mat * sizeof(size_t) * 4);
  cuda_memcpy(gpu_mat, mat, num_mat * sizeof(size_t) * 4, CUDA_HOST_TO_DEV);

  /* 8 bytes contains 4 pixels */
  iw = width / 2;
  ih = height;
  dim3 dimBlock(64, 8);
  dim3 dimGrid(iw / dimBlock.x, ih / dimBlock.y);

  dev_yuv422_single_mask<<<dimGrid, dimBlock, 0, stream>>>(width, height, yuvbuffer, (size_t *)gpu_mat, num_mat);

  cuda_free(gpu_mat);

  return 0;
}


static int host_yuvn12_single_mask(uint8_t *yuvbuffer,
                              size_t width, size_t height, cuda_mat_t *mat, size_t num_mat,
                              cudaStream_t stream) {
  size_t iw, ih;
  if (NULL == yuvbuffer) {
    LOGCUDA("%s invalid param\n", __func__);
    return -1;
  }
  if(num_mat > 100){
    LOGCUDA("illegal mat num %zu", num_mat);
    return -1;
  }
  size_t *gpu_mat = (size_t*)cuda_malloc(num_mat * sizeof(size_t) * 4);
  cuda_memcpy(gpu_mat, mat, num_mat * sizeof(size_t) * 4, CUDA_HOST_TO_DEV);

  iw = width;
  ih = height;

  dim3 dimBlock(64, 8);
  dim3 dimGrid((iw + dimBlock.x - 1) / dimBlock.x,
               (ih + dimBlock.y - 1) / dimBlock.y);
  dev_yuvn12_single_mask<<<dimGrid, dimBlock, 0, stream>>>(width, height, yuvbuffer, (size_t *)gpu_mat, num_mat);
  cuda_free(gpu_mat);

  return 0;
}


extern "C" int cuda_YUVMASK(cuda_image_handle_t *cuda_handle_image, cuda_mat_t *mat, size_t num_mat) {
  int error = -1;
  uint32_t stream_idx;
  cuda_image_cmd_e image_cmd;
  size_t width, height;
  uint8_t *srcbuffer;
  std::vector<unsigned int> results;


  if (NULL == cuda_handle_image) {
    LOGCUDA("%s: invalid param\n", __func__);
    goto failed;
  }
  srcbuffer = (uint8_t *)cuda_handle_image->imagein->plan_buffer[0];
  width = cuda_handle_image->imagein->width;
  height = cuda_handle_image->imagein->height;
  image_cmd = cuda_handle_image->image_cmd;

  stream_idx = cuda_handle_image->stream_idx;
  if (stream_idx > (MAX_STREAM_COUNT - 1)) {
    LOGCUDA("%s: exceed valid stream count\n", __func__);
    goto failed;
  }

  switch (image_cmd) {
    case CUDA_IMG_YUV422:
      error = host_yuv422_single_mask(srcbuffer, width, height, mat, num_mat, 
          cuda_stream_array[stream_idx]);
      if (error) {
        LOGCUDA("%s: host_yuv422_single_mask err\n", __func__);
        goto failed;
      }
      break;

    case CUDA_IMG_UVY422:
      break;

    case CUDA_IMG_YUV420PITCH:
      break;

    case CUDA_IMG_YUV420SPPITCH:
      break;

    case CUDA_IMG_YUV420SP:
      error = host_yuvn12_single_mask(srcbuffer, width, height, mat, num_mat, 
          cuda_stream_array[stream_idx]);
      if (error) {
        LOGCUDA("%s: host_yuvn12_single_mask err\n", __func__);
        goto failed;
      }
      break;

    case CUDA_IMG_I420:
      break;

    default:
      LOGCUDA("%s: invalid image cmd\n", __func__);
      goto failed;
  }

  return 0;

failed:
  return -1;
}

///////////////////////////////////////////host api----init or
///config//////////////////////////////////////////////
extern "C" int cuda_get_dev_count(uint32_t *dev_count) {
  cudaError_t error;

  if (NULL == dev_count) {
    LOGCUDA("%s: invalid param\n", __func__);
    return -1;
  }

  error = cudaGetDeviceCount((int *)dev_count);
  if (error != cudaSuccess) {
    LOGCUDA("%s: cudaGetDeviceCount returned %d\n-> %s\n", __func__, (int)error,
            cudaGetErrorString(error));
    return -1;
  }

  return 0;
}

extern "C" int cuda_get_dev_info(cuda_dev_info_t *cuda_dev_info) {
  int i;
  cudaError_t error;
  uint32_t deviceCount;
  int attrValue = 0;
  int driverVersion = 0, runtimeVersion = 0;

  if (NULL == cuda_dev_info) {
    LOGCUDA("%s: invalid param\n", __func__);
    return -1;
  }

  error = cudaGetDeviceCount((int *)&deviceCount);
  if (error != cudaSuccess) {
    LOGCUDA("%s: cudaGetDeviceCount returned %d\n-> %s\n", __func__, (int)error,
            cudaGetErrorString(error));
    return -1;
  }
  cuda_dev_info->device_count = deviceCount;

  for (i = 0; i < deviceCount; i++) {
    checkCudaErrors(cudaSetDevice(i));
    
    checkCudaErrors(cudaDeviceGetAttribute(&attrValue, cudaDevAttrUnifiedAddressing, i));
    cuda_dev_info->cuda_dev_property[i].device_count_with_unified_memory = attrValue;

    checkCudaErrors(cudaDriverGetVersion(&driverVersion));
    checkCudaErrors(cudaRuntimeGetVersion(&runtimeVersion));
    cuda_dev_info->driver_version = driverVersion;
    cuda_dev_info->runtime_version = runtimeVersion;
  }

  return 0;
}

extern "C" int cuda_dev_init(void) {
  if (is_init) return 0;

  cudaError_t error;
  int deviceCount;
  cuda_dev_ctrl_t *pcuda_dev_ctrl = &g_cuda_dev_ctrl;

  pcuda_dev_ctrl->debug_info = CUDA_DEBUG_NULL;

  error = cudaGetDeviceCount(&deviceCount);
  if (error != cudaSuccess) {
    LOGCUDA("%s: cudaGetDeviceCount returned %d\n-> %s\n", __func__, (int)error,
            cudaGetErrorString(error));
    return -1;
  }

  if (!deviceCount) {
    LOGCUDA("%s: no gpu...\n", __func__);
    return -1;
  }

  pthread_mutex_init(&pcuda_dev_ctrl->mutex, NULL);
  pcuda_dev_ctrl->device_count = deviceCount;
  pcuda_dev_ctrl->stream_count = 0;

  for (uint32_t i = 0; i < MAX_STREAM_COUNT; i++) {
    cuda_stream_create(i);
  }

  is_init = 1;

  return 0;
}

extern "C" int cuda_dev_uninit(void) {
  cuda_dev_ctrl_t *pcuda_dev_ctrl = &g_cuda_dev_ctrl;

  pcuda_dev_ctrl->stream_count = 0;

  return 0;
}

extern "C" int cuda_dev_check(uint32_t *devidx) {
  cudaError_t error;

  error = cudaGetDevice((int *)devidx);
  if (error != cudaSuccess) {
    LOGCUDA("%s: err, returned %d\n-> %s\n", __func__, (int)error,
            cudaGetErrorString(error));
    return -1;
  }

  return 0;
}

extern "C" int cuda_dev_select(uint32_t devidx) {
  cudaError_t error;
  cuda_dev_ctrl_t *pcuda_dev_ctrl = &g_cuda_dev_ctrl;

  if (devidx > (pcuda_dev_ctrl->device_count - 1)) {
    LOGCUDA("%s: invalid devidx...\n", __func__);
    return -1;
  }

  error = cudaSetDevice(devidx);
  if (error != cudaSuccess) {
    LOGCUDA("%s: err, returned %d\n-> %s\n", __func__, (int)error,
            cudaGetErrorString(error));
    return -1;
  }

  return 0;
}

extern "C" int cuda_stream_create(uint32_t streamidx) {
  unsigned int cnt;
  cuda_dev_ctrl_t *pcuda_dev_ctrl = &g_cuda_dev_ctrl;

  pthread_mutex_lock(&pcuda_dev_ctrl->mutex);
  cnt = pcuda_dev_ctrl->stream_count;
  if (cnt > (MAX_STREAM_COUNT - 1)) {
    LOGCUDA("%s: exceed valid stream count\n", __func__);
    pthread_mutex_unlock(&pcuda_dev_ctrl->mutex);
    return -1;
  }
  checkCudaErrors(cudaStreamCreateWithFlags(&cuda_stream_array[cnt], cudaStreamNonBlocking));
  cnt++;
  pcuda_dev_ctrl->stream_count = cnt;
  pthread_mutex_unlock(&pcuda_dev_ctrl->mutex);

  return 0;
}

extern "C" int cuda_stream_destroy(void) {
  unsigned int i, cnt;
  cuda_dev_ctrl_t *pcuda_dev_ctrl = &g_cuda_dev_ctrl;

  cnt = pcuda_dev_ctrl->stream_count;
  for (i = 0; i < cnt; i++) {
    checkCudaErrors(cudaStreamDestroy(cuda_stream_array[i]));
  }

  return 0;
}

extern "C" int cuda_stream_sync(uint32_t streamidx) {
  cudaError_t error;
  unsigned int cnt;
  cuda_dev_ctrl_t *pcuda_dev_ctrl = &g_cuda_dev_ctrl;

  cnt = pcuda_dev_ctrl->stream_count;
  if (streamidx > (cnt - 1)) {
    LOGCUDA("%s: exceed valid stream count\n", __func__);
    goto failed;
  }

  error = cudaStreamSynchronize(cuda_stream_array[streamidx]);
  if (error) {
    LOGCUDA("%s:\n err: %d %s\n\n", __func__, error, cudaGetErrorString(error));
    goto failed;
  }

  return 0;
failed:
  return -1;
}

extern "C" void *cuda_malloc_unified(size_t size, cuda_memory_type_e type) {
  void *buffer = NULL;

  switch (type) {
    case CUDA_HOST:
      checkCudaErrors(cudaMallocHost((void **)&buffer, size));
      if (buffer) {
        return buffer;
      }
      break;
    case CUDA_ATTACH_HOST:
      checkCudaErrors(
          cudaHostAlloc((void **)&buffer, size, cudaMemAttachHost));
      if (buffer) {
        return buffer;
      }
      break;
    case CUDA_ATTACH_GOLABLE:
    default:
      checkCudaErrors(cudaHostAlloc((void **)&buffer, size, cudaMemAttachHost));
      if (buffer) {
        return buffer;
      }
      break;
  }

  return NULL;
}

extern "C" void *cuda_malloc(size_t size) {
  void *buffer = NULL;

  checkCudaErrors(cudaMalloc((void **)&buffer, size));
  if (buffer) {
    return buffer;
  }

  return NULL;
}

extern "C" void *cuda_memcpy(void *dst, void *src, size_t size,
                             cuda_memcpy_direct_e dir) {
  switch (dir) {
    case CUDA_DEV_TO_HOST:
      checkCudaErrors(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
                                      cuda_stream_array[0]));
      checkCudaErrors(cudaStreamSynchronize(cuda_stream_array[0]));
      break;
    case CUDA_HOST_TO_DEV:
      checkCudaErrors(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                                      cuda_stream_array[0]));
      checkCudaErrors(cudaStreamSynchronize(cuda_stream_array[0]));
      break;
    case CUDA_DEV_TO_DEV:
      checkCudaErrors(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice,
                                      cuda_stream_array[0]));
      checkCudaErrors(cudaStreamSynchronize(cuda_stream_array[0]));
      break;
    default:
      return NULL;
  }

  return dst;
}

extern "C" void *cuda_memcpy_async(void *dst, void *src, size_t size,
                                   cuda_memcpy_direct_e dir, uint32_t idx) {
  unsigned int cnt;
  cuda_dev_ctrl_t *pcuda_dev_ctrl = &g_cuda_dev_ctrl;

  cnt = pcuda_dev_ctrl->stream_count;
  if (idx > (cnt - 1)) {
    LOGCUDA("%s: exceed valid stream count\n", __func__);
    return NULL;
  }

  switch (dir) {
    case CUDA_DEV_TO_HOST:
      checkCudaErrors(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
                                      cuda_stream_array[idx]));
      break;
    case CUDA_HOST_TO_DEV:
      checkCudaErrors(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                                      cuda_stream_array[idx]));
      break;
    case CUDA_DEV_TO_DEV:
      checkCudaErrors(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice,
                                      cuda_stream_array[idx]));
      break;
    default:
      return NULL;
  }

  return dst;
}

extern "C" void *cuda_memcpy2D(void *dst, size_t dpitch, const void *src,
                               size_t spitch, size_t width, size_t height,
                               cuda_memcpy_direct_e dir) {
  switch (dir) {
    case CUDA_DEV_TO_HOST:
      checkCudaErrors(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                        cudaMemcpyDeviceToHost,
                                        cuda_stream_array[0]));
      checkCudaErrors(cudaStreamSynchronize(cuda_stream_array[0]));
      break;
    case CUDA_HOST_TO_DEV:
      checkCudaErrors(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                        cudaMemcpyHostToDevice,
                                        cuda_stream_array[0]));
      checkCudaErrors(cudaStreamSynchronize(cuda_stream_array[0]));
      break;
    case CUDA_DEV_TO_DEV:
      checkCudaErrors(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                        cudaMemcpyDeviceToDevice,
                                        cuda_stream_array[0]));
      checkCudaErrors(cudaStreamSynchronize(cuda_stream_array[0]));  
      break;
    default:
      return NULL;
  }

  return dst;
}

extern "C" void *cuda_memcpy2D_async(void *dst, size_t dpitch, const void *src,
                                     size_t spitch, size_t width, size_t height,
                                     cuda_memcpy_direct_e dir, uint32_t idx) {
  unsigned int cnt;
  cuda_dev_ctrl_t *pcuda_dev_ctrl = &g_cuda_dev_ctrl;

  cnt = pcuda_dev_ctrl->stream_count;
  if (idx > (cnt - 1)) {
    LOGCUDA("%s: exceed valid stream count\n", __func__);
    return NULL;
  }

  switch (dir) {
    case CUDA_DEV_TO_HOST:
      checkCudaErrors(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                        cudaMemcpyDeviceToHost,
                                        cuda_stream_array[idx]));
      break;
    case CUDA_HOST_TO_DEV:
      checkCudaErrors(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                        cudaMemcpyHostToDevice,
                                        cuda_stream_array[idx]));
      break;
    case CUDA_DEV_TO_DEV:
      checkCudaErrors(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                        cudaMemcpyDeviceToDevice,
                                        cuda_stream_array[idx]));
      break;
    default:
      return NULL;
  }

  return dst;
}

extern "C" void cuda_free(void *buffer) {
  if (buffer) {
    checkCudaErrors(cudaFree(buffer));
  }
}

extern "C" void cuda_debug(int32_t debg_info) {
  cuda_dev_ctrl_t *pcuda_dev_ctrl = &g_cuda_dev_ctrl;

  pcuda_dev_ctrl->debug_info = debg_info;
}
