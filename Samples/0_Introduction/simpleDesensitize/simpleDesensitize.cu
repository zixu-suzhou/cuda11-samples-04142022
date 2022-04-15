/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample demonstrates how use picture cut in CUDA
 *
 * This sample takes an input yuv image (test.yuv) and generates
 * an output bgr image (test.bgr).  This CUDA kernel performs
 * a simple multi object (sensitive object) cut from source pic .
 */

// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <cmath>
#include <fstream>
#include <iostream>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>  // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>  // helper functions for CUDA error check

#include "cuda_image.h"

static const char *sampleName = "simpleDesensitize";

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

extern "C" void computeGold(float *reference, float *idata,
                            const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf("%s starting...\n", sampleName);

  runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv) {
  if (0 != cuda_dev_init()) {
    printf("FATAL: CUDA device init failed\n");
    return;
  }
  cuda_dev_info_t cuda_dev_info_;
  if (0 != cuda_get_dev_info(&cuda_dev_info_)) {
    printf("FATAL: CUDA device get info failed\n");
    return;
  }

  int yuv_width = 2896;
  int yuv_height = 1876;
  int yuv_image_size = yuv_width * yuv_height * 3 / 2;
  uint8_t *yuv_gpu_buf;
  uint8_t *yuv_cpu_buf;

  yuv_gpu_buf = (uint8_t *)cuda_malloc(yuv_image_size + 2560);
#if 0
  yuv_cpu_buf = (uint8_t *)cuda_malloc_unified(yuv_image_size + 2560, CUDA_HOST);
#else
  yuv_cpu_buf = (uint8_t *)malloc(yuv_image_size + 2560);
#endif
  if (!yuv_cpu_buf || !yuv_gpu_buf) {
    printf("error while malloc GPU buffers!\n");
    return;
  }

  FILE *fp = NULL;
  fp = fopen("./data/yuvnv12.yuv", "rb");
  if (!fp) {
    printf("FATAL: fail to open yuv file\n");
    return;
  }

  fread(yuv_cpu_buf, 1, yuv_image_size, fp);
  fclose(fp);
  cuda_memcpy(yuv_gpu_buf, yuv_cpu_buf, yuv_image_size, CUDA_HOST_TO_DEV);

  int ret = 0;
  struct timeval time;
  long long int microSeconds, oldmicroSeconds;
  cuda_image_handle_t cuda_image_convert;
  cuda_image_t cuda_yuv_image;

  cuda_yuv_image.width = yuv_width;
  cuda_yuv_image.height = yuv_height;

  cuda_yuv_image.plan_buffer[0] = yuv_gpu_buf;
  cuda_yuv_image.plan_buffer[1] =
      cuda_yuv_image.plan_buffer[0] + yuv_width * yuv_height;
  cuda_yuv_image.plan_buffer[2] =
      cuda_yuv_image.plan_buffer[1] + yuv_width * yuv_height / 4;

  cuda_yuv_image.plan_pitch[0] = yuv_width * yuv_height;
  cuda_yuv_image.plan_pitch[1] = yuv_width * yuv_height / 4;
  cuda_yuv_image.plan_pitch[2] = yuv_width * yuv_height / 4;
  cuda_image_convert.image_cmd = CUDA_IMG_YUV420SP;

  cuda_image_convert.imagein = &cuda_yuv_image;
  cuda_image_convert.imageout = &cuda_yuv_image;
  cuda_image_convert.stream_idx = 0;

  gettimeofday(&time, NULL);
  oldmicroSeconds = 1000000 * time.tv_sec + time.tv_usec;

  cuda_mat_t mat[4] = {
    {300, 300, 50, 50}, 
    {500, 500, 100, 100}, 
    {1000, 1000, 200, 200}, 
    {2000, 1500, 300, 300}};
  size_t num_mat = 4;
  ret = cuda_YUVMASK(&cuda_image_convert, mat, num_mat);
  if(ret){
    printf("cuda_YUVMASK failed with ret=%d, output saving will continue\n", ret);
  }

  gettimeofday(&time, NULL);
  microSeconds = 1000000 * time.tv_sec + time.tv_usec;
  printf("yuvmask time used: %lld\n", microSeconds - oldmicroSeconds);

  cuda_memcpy(yuv_cpu_buf, yuv_gpu_buf, yuv_image_size, CUDA_DEV_TO_HOST);
  fp = fopen("./data/output_nv12.yuv", "wb");
  if (!fp) {
    printf("failed to open file for saveing output image\n");
    return;
  }

  fwrite(yuv_cpu_buf, 1, yuv_image_size, fp);
  fclose(fp);

  cuda_free(yuv_gpu_buf);
  free(yuv_cpu_buf);



  yuv_width = 1280;
  yuv_height = 960;
  yuv_image_size = yuv_width * yuv_height * 2;

  yuv_gpu_buf = (uint8_t *)cuda_malloc(yuv_image_size + 2560);
#if 0
  yuv_cpu_buf = (uint8_t *)cuda_malloc_unified(yuv_image_size + 2560, CUDA_HOST);
#else
  yuv_cpu_buf = (uint8_t *)malloc(yuv_image_size + 2560);
#endif
  if (!yuv_cpu_buf || !yuv_gpu_buf) {
    printf("error while malloc GPU buffers!\n");
    return;
  }

  fp = NULL;
  fp = fopen("./data/yuv422.yuv", "rb");
  if (!fp) {
    printf("FATAL: fail to open yuv file\n");
    return;
  }

  fread(yuv_cpu_buf, 1, yuv_image_size, fp);
  fclose(fp);
  cuda_memcpy(yuv_gpu_buf, yuv_cpu_buf, yuv_image_size, CUDA_HOST_TO_DEV);

  ret = 0;

  cuda_yuv_image.width = yuv_width;
  cuda_yuv_image.height = yuv_height;

  cuda_yuv_image.plan_buffer[0] = yuv_gpu_buf;
  cuda_yuv_image.plan_buffer[1] =
      cuda_yuv_image.plan_buffer[0] + yuv_width * yuv_height;
  cuda_yuv_image.plan_buffer[2] =
      cuda_yuv_image.plan_buffer[1] + yuv_width * yuv_height / 4;

  cuda_yuv_image.plan_pitch[0] = yuv_width * yuv_height;
  cuda_yuv_image.plan_pitch[1] = yuv_width * yuv_height / 4;
  cuda_yuv_image.plan_pitch[2] = yuv_width * yuv_height / 4;
  cuda_image_convert.image_cmd = CUDA_IMG_YUV422;

  cuda_image_convert.imagein = &cuda_yuv_image;
  cuda_image_convert.imageout = &cuda_yuv_image;
  cuda_image_convert.stream_idx = 0;

  gettimeofday(&time, NULL);
  oldmicroSeconds = 1000000 * time.tv_sec + time.tv_usec;

  num_mat = 2;
  ret = cuda_YUVMASK(&cuda_image_convert, mat, num_mat);
  if(ret){
    printf("cuda_YUVMASK failed with ret=%d, output saving will continue\n", ret);
  }

  gettimeofday(&time, NULL);
  microSeconds = 1000000 * time.tv_sec + time.tv_usec;
  printf("yuvmask time used: %lld\n", microSeconds - oldmicroSeconds);

  cuda_memcpy(yuv_cpu_buf, yuv_gpu_buf, yuv_image_size, CUDA_DEV_TO_HOST);
  fp = fopen("./data/output_422.yuv", "wb");
  if (!fp) {
    printf("failed to open file for saveing output image\n");
    return;
  }

  fwrite(yuv_cpu_buf, 1, yuv_image_size, fp);
  fclose(fp);

  cuda_free(yuv_gpu_buf);
  free(yuv_cpu_buf);


  return;

}

