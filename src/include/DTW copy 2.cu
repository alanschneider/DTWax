#ifndef FULLDTW
#define FULLDTW

#include "common.hpp"
#include "datatypes.hpp"
#include <cooperative_groups.h>

#ifdef NV_DEBUG
#define REG_ID (SEGMENT_SIZE - 1)
#endif

#ifdef FP16 // FP16 definitions
#include <cuda_fp16.h>
#endif

namespace cg = cooperative_groups;

#define ALL 0xFFFFFFFF

#ifndef NO_REF_DEL
#define COST_FUNCTION(q, r1, l, t, d)                                          \
  FMA(FMA(SUB(r1, q), SUB(r1, q), FLOAT2HALF2(0.0f)), FLOAT2HALF2(1.0f),       \
      FIND_MIN(l, FIND_MIN(t, d)))                                             \
#else // assuming there are no reference
#define COST_FUNCTION(q, r1, l, t, d)                                          \
  FMA(FMA(SUB(r1, q), SUB(r1, q), FLOAT2HALF2(0.0f)), FLOAT2HALF2(1.0f),       \
      FIND_MIN(t, d))
#endif

///////////////////////////////////////////////////////////////////////////////
// compute_segment
//    Computes segments of the sDTW matrix
//
__device__ __forceinline__ void
compute_segment(int       &wave,
                const int &thread_id,
                float     &query_val,
                float     (&ref_coeff1)[SEGMENT_SIZE],
                float     &penalty_left,
                float     (&penalty_here)[SEGMENT_SIZE],
                float     &penalty_diag,
                float     (&penalty_temp)[2],
                int       query_batch)
{
  /* calculate SEGMENT_SIZE cells */
  penalty_temp[0] = penalty_here[0];

  if ((thread_id != (wave - 1)) || (query_batch)) {
    penalty_here[0] = COST_FUNCTION(query_val,
                                    ref_coeff1[0],
                                    penalty_left,
                                    penalty_here[0],
                                    penalty_diag);

    for (int i = 1; i < SEGMENT_SIZE - 2; i += 2) {
      penalty_temp[1] = penalty_here[i];
      penalty_here[i] = COST_FUNCTION(query_val,
                                      ref_coeff1[i],
                                      penalty_here[i - 1],
                                      penalty_here[i],
                                      penalty_temp[0]);

      penalty_temp[0]     = penalty_here[i + 1];
      penalty_here[i + 1] = COST_FUNCTION(query_val,
                                          ref_coeff1[i + 1],
                                          penalty_here[i],
                                          penalty_here[i + 1],
                                          penalty_temp[1]);
    }
    penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(query_val,
                                                   ref_coeff1[SEGMENT_SIZE - 1],
                                                   penalty_here[SEGMENT_SIZE - 2],
                                                   penalty_here[SEGMENT_SIZE - 1],
                                                   penalty_temp[0]);
  }
  else {
    penalty_here[0] = COST_FUNCTION(query_val,
                                    ref_coeff1[0],
                                    penalty_left,
                                    FLOAT2HALF2(0.0f),
                                    penalty_diag);

    for (int i = 1; i < SEGMENT_SIZE - 2; i += 2) {
      penalty_temp[1] = penalty_here[i];
      penalty_here[i] = COST_FUNCTION(query_val,
                                      ref_coeff1[i],
                                      penalty_here[i - 1],
                                      FLOAT2HALF2(0.0f),
                                      FLOAT2HALF2(0.0f));

      penalty_temp[0]     = penalty_here[i + 1];
      penalty_here[i + 1] = COST_FUNCTION(query_val,
                                          ref_coeff1[i + 1],
                                          penalty_here[i],
                                          FLOAT2HALF2(0.0f),
                                          FLOAT2HALF2(0.0f));
    }
    penalty_here[SEGMENT_SIZE - 1] = COST_FUNCTION(query_val,
                                                   ref_coeff1[SEGMENT_SIZE - 1],
                                                   penalty_here[SEGMENT_SIZE - 2],
                                                   FLOAT2HALF2(0.0f),
                                                   FLOAT2HALF2(0.0f));
  }
}

///////////////////////////////////////////////////////////////////////////////
// DTW
//    Subsequence DTW
//
__global__ void DTW(float* ref,
                    float* query,
                    float* dist,
                    int    num_entries,
                    float  thresh,
                    float* device_last_row)
{
  __shared__ float penalty_last_col[PREFIX_LEN];
  const int        block_id         = blockIdx.x;
  const int        thread_id        = threadIdx.x;
  float            penalty_temp[2];
  float            min_segment      = INFINITY; // finds min of segment for sDTW
  float            last_col_penalty_shuffled;   // used to store last col of matrix

  /* each thread computes SEGMENT_SIZE adjacent cells, get corresponding sig
   * values */
  float ref_coeff1[SEGMENT_SIZE];

/* load next WARP_SIZE query values from memory into new_query_val buffer */
  for (int query_batch = 0; query_batch < QUERY_BATCH; query_batch++) {
    float penalty_left               = INFINITY;
    float penalty_diag               = INFINITY;
    float penalty_here[SEGMENT_SIZE] = { 0 };
    float query_val                  = INFINITY;
    float new_query_val              = query[(block_id * QUERY_LEN) + (query_batch * PREFIX_LEN) + thread_id];
    
    /* initialize first thread's chunk */
    if (thread_id == 0) {
      query_val = new_query_val;
    }

    new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

    for (int i = 0; i < SEGMENT_SIZE; i++) {
      ref_coeff1[i] = ref[thread_id + i * WARP_SIZE];

      if (query_batch > 0)
        penalty_here[i] = device_last_row[block_id * REF_LEN + thread_id + i * WARP_SIZE];
    }

    /* calculate full matrix in wavefront parallel manner, multiple cells per
     * thread */
    for (int wave = 1; wave <= NUM_WAVES; wave++) {
      if (query_batch == (QUERY_BATCH - 1))
        min_segment = __shfl_up_sync((ALL), min_segment, 1);

      if (((wave - PREFIX_LEN) <= thread_id) &&
          (thread_id <= (wave - 1))) // HS: block cells that have completed from further
        compute_segment(wave,
                        thread_id,
                        query_val,
                        ref_coeff1,
                        penalty_left,
                        penalty_here,
                        penalty_diag,
                        penalty_temp,
                        query_batch);

      /* new_query_val buffer is empty, reload */
      if ((wave & (WARP_SIZE_MINUS_ONE)) == 0) {
        new_query_val = query[(block_id * QUERY_LEN) +
                              (query_batch * PREFIX_LEN) + wave + thread_id];
      }

      query_val    = __shfl_up_sync(ALL, query_val, 1); // pass next query_value to each thread
      penalty_diag = penalty_left;                      // transfer border cell info
      penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1], 1);

      if (thread_id == 0) {
        query_val    = new_query_val;
        penalty_left = FLOAT2HALF2(INFINITY);
      }
      new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

      // Find min of segment and then shuffle up for sDTW
      //
      if ((wave >= PREFIX_LEN) && (query_batch == (QUERY_BATCH - 1))) {
        for (int i = 0; i < SEGMENT_SIZE; i++) {
          min_segment = FIND_MIN(min_segment, penalty_here[i]);
        }
      }
    }
  }

  if (thread_id == WARP_SIZE_MINUS_ONE) {
    dist[block_id] = min_segment;
  }
  return;
}

#endif