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
template <typename idx_t, typename val_t>
__device__ __forceinline__ void
compute_segment(idxt        &wave,
                const idx_t &thread_id,
                val_t       &query_val,
                val_t       (&ref_coeff1)[SEGMENT_SIZE],
                val_t       &penalty_left,
                val_t       (&penalty_here)[SEGMENT_SIZE],
                val_t       &penalty_diag,
                val_t       (&penalty_temp)[2],
                idxt        query_batch)
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
template <typename idx_t, typename val_t>
__global__ void DTW(reference_coefficients* ref,
                    val_t*                  query,
                    val_t*                  dist,
                    idx_t                   num_entries,
                    val_t                   thresh,
                    val_t*                  device_last_row) {
  __shared__ val_t penalty_last_col[PREFIX_LEN]; ////RBD: have to chnge this
  const idx_t      block_id         = blockIdx.x;
  const idx_t      thread_id        = threadIdx.x;
  val_t            penalty_temp[2];
  val_t            min_segment      = FLOAT2HALF2(INFINITY); // finds min of segment for sDTW
  val_t            last_col_penalty_shuffled;           // used to store last col of matrix

  /* each thread computes SEGMENT_SIZE adjacent cells, get corresponding sig
   * values */
  val_t ref_coeff1[SEGMENT_SIZE];

/* load next WARP_SIZE query values from memory into new_query_val buffer */
#pragma unroll
  for (idxt query_batch = 0; query_batch < QUERY_BATCH; query_batch++) {
    val_t penalty_left               = FLOAT2HALF2(INFINITY);
    val_t penalty_diag               = FLOAT2HALF2(INFINITY);
    val_t penalty_here[SEGMENT_SIZE] = { FLOAT2HALF2(0) };
    val_t query_val                  = FLOAT2HALF2(INFINITY);
    val_t new_query_val              = query[(block_id * QUERY_LEN) + (query_batch * PREFIX_LEN) + thread_id];
    
    /* initialize first thread's chunk */
    if (thread_id == 0) {
      query_val = new_query_val;
    }

    new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

#pragma unroll
    for (idxt i = 0; i < SEGMENT_SIZE; i++) {
      ref_coeff1[i] = ref[thread_id + i * WARP_SIZE].coeff1;

      if (query_batch > 0)
        penalty_here[i] = device_last_row[block_id * REF_LEN + thread_id + i * WARP_SIZE];
    }

    /* calculate full matrix in wavefront parallel manner, multiple cells per
     * thread */
    for (idxt wave = 1; wave <= NUM_WAVES; wave++) {
      if (query_batch == (QUERY_BATCH - 1))
        min_segment = __shfl_up_sync((ALL), min_segment, 1);

      if (((wave - PREFIX_LEN) <= thread_id) &&
          (thread_id <= (wave - 1))) // HS: block cells that have completed from further
        compute_segment<idx_t, val_t>(wave,
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

#if REF_BATCH > 1
      last_col_penalty_shuffled = __shfl_down_sync(ALL, last_col_penalty_shuffled, 1);
      if (thread_id == WARP_SIZE_MINUS_ONE)
        last_col_penalty_shuffled = penalty_here[RESULT_REG];
      if ((wave >= TWICE_WARP_SIZE_MINUS_ONE) &&
          ((wave & WARP_SIZE_MINUS_ONE) == WARP_SIZE_MINUS_ONE))
      { // HS
        penalty_last_col[(wave - TWICE_WARP_SIZE_MINUS_ONE) + thread_id] = last_col_penalty_shuffled;
      }
      else if ((wave >= NUM_WAVES_BY_WARP_SIZE) &&
               (thread_id == WARP_SIZE_MINUS_ONE))
      {
        penalty_last_col[wave - WARP_SIZE] = penalty_here[RESULT_REG];
      }

#endif
      // Find min of segment and then shuffle up for sDTW
      //
      if ((wave >= PREFIX_LEN) && (query_batch == (QUERY_BATCH - 1))) {
        for (idxt i = 0; i < SEGMENT_SIZE; i++) {
          min_segment = FIND_MIN(min_segment, penalty_here[i]);
        }
      }
    }

// write last row to smem
#if QUERY_BATCH > 1
    for (idxt i = 0; i < SEGMENT_SIZE; i++) {
      device_last_row[block_id * REF_LEN + thread_id + i * WARP_SIZE] = penalty_here[i];
    }
#endif

#if REF_BATCH > 1
    // For all ref batches > 0
    //
    for (idxt ref_batch = 1; ref_batch < REF_BATCH_MINUS_ONE; ref_batch++) {

      if (query_batch == (QUERY_BATCH - 1))
        min_segment = __shfl_down_sync((ALL), min_segment, 31);

      /* initialize penalties */
      penalty_diag = FLOAT2HALF2(INFINITY);
      penalty_left = FLOAT2HALF2(INFINITY);

      if (query_batch == 0) {
        for (auto i = 0; i < SEGMENT_SIZE; i++)
          penalty_here[i] = FLOAT2HALF2(0);

      }
      else {
        for (auto i = 0; i < SEGMENT_SIZE; i++)
          penalty_here[i] = device_last_row[block_id * REF_LEN + ref_batch * REF_TILE_SIZE + thread_id + i * WARP_SIZE];
      }

      /* load next WARP_SIZE query values from memory into new_query_val buffer
       */
      query_val     = FLOAT2HALF2(INFINITY);
      new_query_val = query[(block_id * QUERY_LEN) + (query_batch * PREFIX_LEN) + thread_id];

      /* initialize first thread's chunk */
      if (thread_id == 0) {
        query_val    = new_query_val;
        penalty_left = penalty_last_col[0];
      }
      new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

      for (idxt i = 0; i < SEGMENT_SIZE; i++) {
        ref_coeff1[i] = ref[ref_batch * (REF_TILE_SIZE) + thread_id + i * WARP_SIZE].coeff1;
      }

      /* calculate full matrix in wavefront parallel manner, multiple cells per
       * thread */
      for (idxt wave = 1; wave <= NUM_WAVES; wave++) {
        if (((wave - PREFIX_LEN) <= thread_id) &&
            (thread_id <= (wave - 1))) // HS: block cells that have completed from further
          compute_segment<idx_t, val_t>(wave,
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
          new_query_val = query[(block_id * QUERY_LEN) + (query_batch * PREFIX_LEN) + wave + thread_id];
        }

        /* pass next query_value to each thread */
        query_val = __shfl_up_sync(ALL, query_val, 1);
        if (thread_id == 0) {
          query_val = new_query_val;
        }

        last_col_penalty_shuffled = __shfl_down_sync(ALL, last_col_penalty_shuffled, 1);

        if (thread_id == WARP_SIZE_MINUS_ONE)
          last_col_penalty_shuffled = penalty_here[RESULT_REG];
        if ((wave >= TWICE_WARP_SIZE_MINUS_ONE) &&
            ((wave & WARP_SIZE_MINUS_ONE) == WARP_SIZE_MINUS_ONE)) // HS
          penalty_last_col[(wave - TWICE_WARP_SIZE_MINUS_ONE) + thread_id] = last_col_penalty_shuffled;
        else if ((wave >= NUM_WAVES_BY_WARP_SIZE) &&
                 (thread_id == WARP_SIZE_MINUS_ONE))
          penalty_last_col[(wave - TWICE_WARP_SIZE_MINUS_ONE)] = penalty_here[RESULT_REG];
        
        new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

        /* transfer border cell info */
        penalty_diag = penalty_left;
        penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1], 1);

        if (thread_id == 0) {
          penalty_left = penalty_last_col[wave];
        }

        // Find min of segment and then shuffle up for sDTW
        if ((wave >= PREFIX_LEN) && (query_batch == (QUERY_BATCH - 1))) {
          for (idxt i = 0; i < SEGMENT_SIZE; i++) {
            min_segment = FIND_MIN(min_segment, penalty_here[i]);
          }
          if (wave != (NUM_WAVES))
            min_segment = __shfl_up_sync((ALL), min_segment, 1);
        }
      }

// write last row to smem
#if QUERY_BATCH > 1
      for (idxt i = 0; i < SEGMENT_SIZE; i++) {
        device_last_row[block_id * REF_LEN + ref_batch * REF_TILE_SIZE +
                        thread_id + i * WARP_SIZE] = penalty_here[i];
      }
#endif
    }

    // Last sub-matrix calculation or ref_batch=REF_BATCH-1
    //
    if (query_batch == (QUERY_BATCH - 1))
      min_segment = __shfl_down_sync((ALL), min_segment, 31);

    /* initialize penalties */
    penalty_diag = FLOAT2HALF2(INFINITY);
    penalty_left = FLOAT2HALF2(INFINITY);

    if (query_batch == 0) {
      for (auto i = 0; i < SEGMENT_SIZE; i++)
        penalty_here[i] = FLOAT2HALF2(0);
    } else {
      for (auto i = 0; i < SEGMENT_SIZE; i++) {
        penalty_here[i] = device_last_row[block_id * REF_LEN + REF_LEN - REF_TILE_SIZE + thread_id + i * WARP_SIZE];
      }
    }

    /* load next WARP_SIZE query values from memory into new_query_val buffer */
    query_val     = FLOAT2HALF2(INFINITY);
    new_query_val = query[(block_id * QUERY_LEN) + (query_batch * PREFIX_LEN) + thread_id];

    /* initialize first thread's chunk */
    if (thread_id == 0) {
      query_val    = new_query_val;
      penalty_left = penalty_last_col[0];
    }
    new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

    for (idxt i = 0; i < SEGMENT_SIZE; i++) {
      ref_coeff1[i] = ref[REF_BATCH_MINUS_ONE * (REF_TILE_SIZE) + thread_id + i * WARP_SIZE].coeff1;
    }

    /* calculate full matrix in wavefront parallel manner, multiple cells per
     * thread */
    for (idxt wave = 1; wave <= NUM_WAVES; wave++) {

      if (((wave - PREFIX_LEN) <= thread_id) &&
          (thread_id <= (wave - 1))) // HS: block cells that have completed from further
        compute_segment<idx_t, val_t>(wave,
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
        new_query_val = query[(block_id * QUERY_LEN) + (query_batch * PREFIX_LEN) + wave + thread_id];
      }

      /* pass next query_value to each thread */
      query_val = __shfl_up_sync(ALL, query_val, 1);
      if (thread_id == 0) {
        query_val = new_query_val;
      }

      new_query_val = __shfl_down_sync(ALL, new_query_val, 1);

      /* transfer border cell info */
      penalty_diag = penalty_left;
      penalty_left = __shfl_up_sync(ALL, penalty_here[SEGMENT_SIZE - 1], 1);

      if (thread_id == 0) {
        penalty_left = penalty_last_col[wave];
      }

      // Find min of segment and then shuffle up for sDTW
      if ((wave >= PREFIX_LEN) && (query_batch == (QUERY_BATCH - 1))) {
        for (idxt i = 0; i < SEGMENT_SIZE; i++) {
          min_segment = FIND_MIN(min_segment, penalty_here[i]);
        }

        if (wave != (NUM_WAVES))
          min_segment = __shfl_up_sync((ALL), min_segment, 1);
      }
    }
    // write last row to smem
#if QUERY_BATCH > 1
    for (idxt i = 0; i < SEGMENT_SIZE; i++) {
      device_last_row[block_id * REF_LEN + REF_LEN - REF_TILE_SIZE + thread_id + i * WARP_SIZE] = penalty_here[i];
    }
#endif

#endif

  } // query_batch loop

  if (thread_id == WARP_SIZE_MINUS_ONE) {
    dist[block_id] = min_segment;
  }
  return;
}

#endif //}