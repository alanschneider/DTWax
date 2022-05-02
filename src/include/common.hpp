#ifndef COMMON_HPP
#define COMON_HPP
#endif

//-------------global datatypes---------------------------//
typedef float value_t;   // data type for values
typedef int64_t index_t; // data type for indices
typedef int8_t label_t;  // data type for label
typedef int idxt;
typedef float raw_t;

//...............global variables..........................//
#ifndef FP16
#define SEGMENT_SIZE 32
#else
#define SEGMENT_SIZE 32
#endif

#define WARP_SIZE 32
#define QUERY_LEN 1024
#define BLOCK_NUM (108 * 32)
#define STREAM_NUM 16

//-----------------derived variables--------------------------//
#define REF_LEN QUERY_LEN
#define NUM_READS (BLOCK_NUM * STREAM_NUM * 30)
#define GROUP_SIZE WARP_SIZE
#define CELLS_PER_THREAD SEGMENT_SIZE

#define REF_BATCH (REF_LEN / (SEGMENT_SIZE * WARP_SIZE))

/* calculate when to stop, and which thread has final result */
#define NUM_WAVES (QUERY_LEN + (REF_LEN - 1) / (CELLS_PER_THREAD * REF_BATCH))
#define RESULT_THREAD_ID (WARP_SIZE - 1)
#define RESULT_REG ((QUERY_LEN - 1) % CELLS_PER_THREAD)
