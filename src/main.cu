#include <assert.h>
#include <cstdint>
#include <iostream>
#include <string>

#include "include/DTW.hpp"
#include "include/binary_IO.hpp"
#include "include/cbf_generator.hpp"
#include "include/common.hpp"
#include "include/hpc_helpers.hpp"

using namespace FullDTW;

//------------------time macros--------------------------//
#define TIMERSTART_CUDA(label)                                                 \
  cudaSetDevice(0);                                                            \
  cudaEvent_t start##label, stop##label;                                       \
  float time##label;                                                           \
  cudaEventCreate(&start##label);                                              \
  cudaEventCreate(&stop##label);                                               \
  cudaEventRecord(start##label, 0);

#define TIMERSTOP_CUDA(label)                                                  \
  cudaSetDevice(0);                                                            \
  cudaEventRecord(stop##label, 0);                                             \
  cudaEventSynchronize(stop##label);                                           \
  cudaEventElapsedTime(&time##label, start##label, stop##label);               \
  std::cout << "TIMING: " << time##label << " ms "                             \
            << ((QUERY_LEN + 1) * (REF_LEN + 1) * num_entries) /               \
                   (time##label * 1e6)                                         \
            << " GCUPS (" << #label << ")" << std::endl;
//..................time macros............................//

//--------------------mem allocate-----///
// int mem_allocate(value_t *host_query,value_t *host_ref,value_t
// *device_query,value_t *device_ref,index_t num_entries,value_t *host_dist,
// value_t *device_dist){
//   //----host mem allocation----------------//
//   cudaMallocHost(&host_query,
//                  sizeof(value_t) * num_entries * QUERY_LEN); /* input */
//   cudaMallocHost(&host_ref, sizeof(value_t) * REF_LEN);      /* input */
//   cudaMallocHost(&host_dist,
//                  sizeof(value_t) * num_entries); /* results */

//   //-------dev mem allocation----------//
//   cudaMalloc(&device_query, sizeof(value_t) * num_entries * QUERY_LEN);

//   cudaMalloc(&device_ref, sizeof(value_t) * num_entries * num_entries);
//   cudaMallocHost(&host_dist, sizeof(value_t) * num_entries * num_entries);
//   cudaMalloc(&device_dist, sizeof(value_t) * num_entries * num_entries);
//   return 1;
// }

int main(int argc, char *argv[]) {

  TIMERSTART(malloc)
  index_t num_entries = BLOCK_NUM; // number of sequences

  /* count total cell updates */
  const value_t CU = QUERY_LEN * REF_LEN * num_entries;
  std::cout << "We are going to process " << CU / 1000000000.0
            << " Giga Cell Updates (GCU)" << std::endl;

  // create host storage and buffers on devices
  value_t *host_query = nullptr,      // time series on CPU
      *host_dist = nullptr,           // distance results on CPU
          *device_query = nullptr,    // time series on GPU
              *device_dist = nullptr, // distance results on GPU
                  *host_ref = nullptr, *device_ref = nullptr;

  cudaMallocHost(&host_query,
                 sizeof(value_t) * num_entries * QUERY_LEN); /* input */
  cudaMallocHost(&host_ref, sizeof(value_t) * REF_LEN);      /* input */
  cudaMallocHost(&host_dist, sizeof(value_t) * num_entries); /* results */

  //-------dev mem allocation----------//
  cudaMalloc(&device_query, sizeof(value_t) * num_entries * QUERY_LEN);

  cudaMalloc(&device_ref, sizeof(value_t) * REF_LEN);
  cudaMalloc(&device_dist, sizeof(value_t) * num_entries);

  CUERR
  TIMERSTOP(malloc)

  /* load data from memory into CPU array, initialize GPU results */
  TIMERSTART(load_data)
  generate_cbf(host_query, QUERY_LEN, num_entries);
  // load_binary(host_dist, QUERY_LEN * num_entries,
  //             "../../../data/kernel/dtw_car.bin");
  cudaMemcpyAsync(device_query, host_query,
                  sizeof(value_t) * QUERY_LEN * num_entries,
                  cudaMemcpyHostToDevice);
  cudaMemcpyAsync(
      device_ref,
      &host_query[QUERY_LEN], //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!//
      sizeof(value_t) * REF_LEN, cudaMemcpyHostToDevice);
  CUERR
  cudaMemsetAsync(device_dist, 0, sizeof(value_t) * num_entries);
  CUERR
  TIMERSTOP(load_data)

  /* perform pairwise DTW computation */
  TIMERSTART_CUDA(computation)
  distances(device_ref, device_query, device_dist, num_entries, (float)0.0);
  CUERR
  TIMERSTOP_CUDA(computation)

  /* copy results to cpu */
  TIMERSTART(save_data)
  cudaMemcpyAsync(host_dist, device_dist, sizeof(value_t) * num_entries,
                  cudaMemcpyDeviceToHost);
  CUERR
  TIMERSTOP(save_data)

#ifdef NV_DEBUG
  /* /1* debug output print *1/ */
  // std::cout << "RESULTS:" << std::endl;
  // for (int i = 0; i < num_entries; i++) {
  for (idxt j = 0; j < num_entries; j++) {
    std::cout << host_dist[j] << " ";
  }
  std::cout << std::endl;
  //}
  std::cout << std::endl;
#endif

  TIMERSTART(free)
  cudaFree(device_dist);
  CUERR
  cudaFree(device_query);
  CUERR
  cudaFree(device_ref);
  CUERR
  cudaFreeHost(host_ref);
  CUERR
  cudaFreeHost(host_query);
  CUERR
  cudaFreeHost(host_dist);
  CUERR
  TIMERSTOP(free)

  return 0;
}
