#ifndef _CUDA_IMAGE_H_
#define _CUDA_IMAGE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#define MAX_DEV_NUM 8

/* cuda-memcpy-direct-value */
typedef enum {
  /* the memory from cpu to gpu */
  CUDA_HOST_TO_DEV = 0X00,
  /* the memory from gpu to cpu */
  CUDA_DEV_TO_HOST,
  /* the memory from gpu to gpu */
  CUDA_DEV_TO_DEV,
} cuda_memcpy_direct_e;

/*cmd-value*/
typedef enum {
  /*yuv422*/
  CUDA_IMG_YUV422 = 0X00,

  CUDA_IMG_UVY422,

  /* YYYYYYYYXXXX..,UUX...,VVX... */
  CUDA_IMG_YUV420PITCH,

  /*YYYYYYYY,UV,UV: NV12*/
  CUDA_IMG_YUV420SP,

  /*YYYYYYYY,UV,UV*/
  CUDA_IMG_YUV420SPPITCH,

  // YYYYYYYY UU VV
  CUDA_IMG_I420
} cuda_image_cmd_e;



typedef struct cuda_dev_property {
  uint32_t device_count_with_unified_memory;
  uint32_t device_count_with_concurrent_managed;
} cuda_dev_property_t;

typedef struct cuda_dev_info {
  /*the global memory size in the gpu */
  uint64_t memory_size;
  /*the driver api version in the gpu */
  uint32_t driver_version;
  /*the the runtime api version in the gpu */
  uint32_t runtime_version;
  /*unified memory not supported on this device, 0 not support, 1 support */
  uint32_t is_unified_memory_supported;

  uint32_t device_count;

  cuda_dev_property_t cuda_dev_property[MAX_DEV_NUM];
} cuda_dev_info_t;

typedef struct cuda_image {
  /*yaddr,uaddr,vaddr*/
  uint8_t *plan_buffer[3];
  /*ypitch,upitch,vpitch*/
  size_t plan_pitch[3];
  /* Image width */
  ssize_t width;
  /* Image height */
  ssize_t height;
} cuda_image_t;

typedef struct cuda_img_roi {
  /* pixel position x */
  ssize_t x;
  /* pixel position x */
  ssize_t y;
  /*img roi width */
  ssize_t width;
  /*img roi height */
  ssize_t height;
} cuda_img_roi_t;


typedef struct cuda_image_handle {
  /*pointer to the image to be handle */
  cuda_image_t *imagein;
  /*pointer to the result image  */
  cuda_image_t *imageout;
  /*indicate which way you want to handle with, refer to cmd-value defined*/
  cuda_image_cmd_e image_cmd;
  /*source imgage roi info */
  cuda_img_roi_t src_img_roi;
  /*index of cuda stream */
  uint32_t stream_idx;
} cuda_image_handle_t;

typedef struct cuda_mat {
  size_t fx0;
  size_t fy0;
  size_t scalex;
  size_t scaley;
} cuda_mat_t;


/* cuda_debug-info-value */
typedef enum {
  /*nothing will be print, unless some error occurs */
  CUDA_DEBUG_NULL = (1 << 0),
  /*debug about image crop test information */
  CUDA_DEBUG_IMG_CROP = (1 << 1),
  /*debug about image crop test information */
  CUDA_DEBUG_IMG_RESIZE = (1 << 2),
} cuda_debug_info_e;

/* cuda_dev-status-value */
typedef enum {
  /* indicate the cuda is ready for handling the image */
  CUDA_READY = 0X00,
  /* indicate the cuda is busy with handling the image */
  CUDA_BUSY,
} cuda_status_e;

/* cuda-memory-type-value */
typedef enum {
  /* Memory can be accessed by any stream on any device */
  CUDA_ATTACH_GOLABLE = 0X00,
  /* Memory cannot be accessed by any stream on any device */
  CUDA_ATTACH_HOST,
  CUDA_HOST,
} cuda_memory_type_e;


/*
 *func: get device count on the current gpu.
 *dev_count: store the device count
 *ret: success return 0, failed return -1
 */
int cuda_get_dev_count(uint32_t *dev_count);

/*
 *func: get information frome the specific device, such as memory size, cuda
 *verson eg. cuda_dev_info: store the parameters about the device, refer to
 *cuda_dev_info_t defined ret: success return 0, failed return -1
 */
int cuda_get_dev_info(cuda_dev_info_t *cuda_dev_info);

/*
 *brief:
    select which device you need.
    you must pass the device idx, which support unified memory on the device if
 you use cuda unified memory.

 *func: select which device you need init
 *id: the device idx.
 *ret: success return 0, failed return -1.
 */
int cuda_dev_init(void);

/*
 *func: select which device you need uninit
 *id: the device idx.
 *ret: success return 0, failed return -1.
 */
int cuda_dev_uninit(void);

/*
 *func: malloc memory unified memory from gpu .
 *ret: success return the address, failed return NULL.
 */
void *cuda_malloc(size_t size);

/*
 *func: malloc memory unified memory from gpu .
 *type: the type of memory you want to malloc, refer to cuda-memory-type-value
 *ret: success return the address, failed return NULL.
 */
void *cuda_malloc_unified(size_t size, cuda_memory_type_e type);

/*
 *func: free memory malloced from gpu.
 */
void cuda_free(void *buffer);


/*
 *func: copy memory between cpu and gpu Synchronously.
 *dst: the destination buffer.
 *src: the source buffer.
 *size: copy size of memory
 *dir: the direct between cpu and gpu, refer to cuda-memcpy-direct-value
 *ret: success return 0, failed return -1.
 */
void *cuda_memcpy(void *dst, void *src, size_t size, cuda_memcpy_direct_e dir);

int cuda_YUVMASK(cuda_image_handle_t *cuda_image, cuda_mat_t *mat, size_t num_mat);

int cuda_dev_check(uint32_t *devidx);

int cuda_dev_select(uint32_t devidx);

/*
 *func: cuda stream create. this api must be called after cuda_dev_init()
 *called. streamidx: pointer to the stream created. ret: success return 0,
 *failed return -1.
 */
int cuda_stream_create(uint32_t streamidx);

/*
 *func: cuda stream destroy. this api destroy all cuda stream created
 *ret: success return 0, failed return -1.
 */
int cuda_stream_destroy(void);
/*
 *func: copy memory between cpu and gpu Asynchronously on the specified cuda
 *stream. dst: the destination buffer. src: the source buffer. size: copy size
 *of memory dir: the direct between cpu and gpu, refer to
 *cuda-memcpy-direct-value idx: the index of cuda stream. ret: success return 0,
 *failed return -1.
 */
void *cuda_memcpy_async(void *dst, void *src, size_t size,
                        cuda_memcpy_direct_e dir, uint32_t idx);

/*
 *func: copy 2Dmemory between cpu and gpu Synchronously.
 *dst: the destination buffer.
 *dpitch: the pitch of the destination buffer.
 *src: the source buffer.
 *spitch: the pitch of the source buffer.
 *width: the source buffer width.
 *height: the source buffer height.
 *dir: the direct between cpu and gpu, refer to cuda-memcpy-direct-value
 *ret: success return 0, failed return -1.
 */
void *cuda_memcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                    size_t width, size_t height, cuda_memcpy_direct_e dir);

/*
 *func: copy 2Dmemory between cpu and gpu Asynchronously on the specified cuda
 *stream. dst: the destination buffer. dpitch: the pitch of the destination
 *buffer. src: the source buffer. spitch: the pitch of the source buffer. width:
 *the source buffer width. height: the source buffer height. dir: the direct
 *between cpu and gpu, refer to cuda-memcpy-direct-value idx: the index of cuda
 *stream. ret: success return 0, failed return -1.
 */
void *cuda_memcpy2D_async(void *dst, size_t dpitch, const void *src,
                          size_t spitch, size_t width, size_t height,
                          cuda_memcpy_direct_e dir, uint32_t idx);

/*
 *func: cuda_debug.
 *debg_info: decide which debug information you want present. refer to
 *cuda_debug-info-value defined above. ret: none.
 */
void cuda_debug(int32_t debg_info);


#ifdef __cplusplus
};
#endif

#endif
