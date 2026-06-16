// Treasure Master — cub sort-based collapse for the GPU wide-merge dedup.
//
// Compiled by nvcc (CUDA Runtime API + cub); linked into the otherwise Driver-API
// `test_cuda` host driver. Step 3 of the wide-merge build: take the per-candidate
// 64-bit fingerprints produced by tm_wide_merge_fp_dump_n_cuda and collapse them by
// sorting (fp, idx) pairs + run-length-encoding the sorted fingerprints. Each run =
// one unique state; the run's first index is the survivor (representative), the run
// length is its multiplicity (how many data values map to it). This replaces the
// compaction path's union-find rep tracking for the sort variant.
//
// ISOLATION: self-contained runtime-API device memory + host-pointer interface, so
// it does NOT touch the Driver-API harness's CUcontext. A host round-trip is used
// for validation/cost de-risking; the production zero-copy path (cub on the
// driver-allocated fp buffer via a shared primary context) is a later optimization.
//
// Build: nvcc -O3 -arch=sm_120 -I<cuda>/targets/x86_64-linux/include/cccl -c

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>

#define WCK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  fprintf(stderr,"[wms] CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); return -1; } } while(0)

__global__ void wms_iota(uint32_t* v, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = i;
}

// survivor_idx[r] = vals_sorted[offset[r]]  (first index of run r)
__global__ void wms_gather_first(const uint32_t* vals_sorted, const uint32_t* offsets,
                                 uint32_t* survivor_idx, uint32_t num_runs) {
    const uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < num_runs) survivor_idx[r] = vals_sorted[offsets[r]];
}

extern "C" {

// Returns 0 on success, -1 on CUDA error. On success:
//   *out_unique         = number of distinct fingerprints (collapsed frontier size)
//   h_survivor_idx[0..U)= first original index of each run (the representatives)
//   h_run_len[0..U)     = run length of each run (multiplicity)
//   *out_sort_ms        = sort+RLE+scan+gather kernel time (ms), copies excluded
// Caller pre-allocates h_survivor_idx and h_run_len with n elements.
int wms_sort_dedup(const uint64_t* h_fps, uint32_t n,
                   uint32_t* h_survivor_idx, uint32_t* h_run_len,
                   uint32_t* out_unique, float* out_sort_ms)
{
    uint64_t *d_kin=nullptr,*d_kout=nullptr; uint32_t *d_vin=nullptr,*d_vout=nullptr;
    WCK(cudaMalloc(&d_kin,  (size_t)n*8));
    WCK(cudaMalloc(&d_kout, (size_t)n*8));
    WCK(cudaMalloc(&d_vin,  (size_t)n*4));
    WCK(cudaMalloc(&d_vout, (size_t)n*4));
    WCK(cudaMemcpy(d_kin, h_fps, (size_t)n*8, cudaMemcpyHostToDevice));
    WCK(cudaMemset(d_vin, 0, (size_t)n*4));
    wms_iota<<<(n+255)/256,256>>>(d_vin, n);

    uint64_t *d_uk=nullptr; uint32_t *d_rl=nullptr,*d_off=nullptr,*d_nruns=nullptr,*d_surv=nullptr;
    WCK(cudaMalloc(&d_uk,  (size_t)n*8));
    WCK(cudaMalloc(&d_rl,  (size_t)n*4));
    WCK(cudaMalloc(&d_off, (size_t)n*4));
    WCK(cudaMalloc(&d_surv,(size_t)n*4));
    WCK(cudaMalloc(&d_nruns, 4));

    void *t_sort=nullptr,*t_rle=nullptr,*t_scan=nullptr; size_t s_sort=0,s_rle=0,s_scan=0;
    cub::DeviceRadixSort::SortPairs(t_sort, s_sort, d_kin,d_kout,d_vin,d_vout,n);
    cub::DeviceRunLengthEncode::Encode(t_rle, s_rle, d_kout, d_uk, d_rl, d_nruns, n);
    cub::DeviceScan::ExclusiveSum(t_scan, s_scan, d_rl, d_off, n);
    WCK(cudaMalloc(&t_sort, s_sort));
    WCK(cudaMalloc(&t_rle,  s_rle));
    WCK(cudaMalloc(&t_scan, s_scan));

    // Warmup pass (untimed) — absorbs cub first-call init so the timed region is
    // apples-to-apples with the global-hash kernel's warmed launch.
    {
        cub::DeviceRadixSort::SortPairs(t_sort, s_sort, d_kin,d_kout,d_vin,d_vout,n);
        uint32_t wruns=0;
        cub::DeviceRunLengthEncode::Encode(t_rle, s_rle, d_kout, d_uk, d_rl, d_nruns, n);
        WCK(cudaMemcpy(&wruns, d_nruns, 4, cudaMemcpyDeviceToHost));
        cub::DeviceScan::ExclusiveSum(t_scan, s_scan, d_rl, d_off, wruns);
        wms_gather_first<<<(wruns+255)/256,256>>>(d_vout, d_off, d_surv, wruns);
        WCK(cudaDeviceSynchronize());
    }

    cudaEvent_t e0,e1; WCK(cudaEventCreate(&e0)); WCK(cudaEventCreate(&e1));
    WCK(cudaEventRecord(e0));
    cub::DeviceRadixSort::SortPairs(t_sort, s_sort, d_kin,d_kout,d_vin,d_vout,n);
    cub::DeviceRunLengthEncode::Encode(t_rle, s_rle, d_kout, d_uk, d_rl, d_nruns, n);
    uint32_t nruns=0;
    WCK(cudaMemcpy(&nruns, d_nruns, 4, cudaMemcpyDeviceToHost));
    cub::DeviceScan::ExclusiveSum(t_scan, s_scan, d_rl, d_off, nruns);
    wms_gather_first<<<(nruns+255)/256,256>>>(d_vout, d_off, d_surv, nruns);
    WCK(cudaEventRecord(e1));
    WCK(cudaEventSynchronize(e1));
    float ms=0; WCK(cudaEventElapsedTime(&ms, e0,e1));

    WCK(cudaMemcpy(h_survivor_idx, d_surv, (size_t)nruns*4, cudaMemcpyDeviceToHost));
    WCK(cudaMemcpy(h_run_len,      d_rl,   (size_t)nruns*4, cudaMemcpyDeviceToHost));
    *out_unique = nruns;
    *out_sort_ms = ms;

    cudaFree(d_kin);cudaFree(d_kout);cudaFree(d_vin);cudaFree(d_vout);
    cudaFree(d_uk);cudaFree(d_rl);cudaFree(d_off);cudaFree(d_surv);cudaFree(d_nruns);
    cudaFree(t_sort);cudaFree(t_rle);cudaFree(t_scan);
    cudaEventDestroy(e0);cudaEventDestroy(e1);
    return 0;
}

} // extern "C"

struct WmsAddU32 { __host__ __device__ uint32_t operator()(uint32_t a, uint32_t b) const { return a + b; } };

// Gather two values per run-start: idx and (already-sorted) — generic gather.
__global__ void wms_gather2(const uint32_t* src, const uint32_t* perm, uint32_t* dst, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[perm[i]];
}

extern "C" {

// Multiplicity-composing collapse for PERIODIC multi-merge. Inputs are the current
// survivors: fingerprint, rep data value (idx_in), and incoming multiplicity (mult_in).
// Groups by fp; per group emits the first rep (idx_out) and the SUM of incoming
// multiplicities (mult_out) — so multiplicity composes correctly across merges.
// Returns U = number of groups. Caller pre-allocates idx_out/mult_out with n.
int wms_sort_dedup_mult(const uint64_t* h_fps, const uint32_t* h_idx_in, const uint32_t* h_mult_in,
                        uint32_t n, uint32_t* h_idx_out, uint32_t* h_mult_out, uint32_t* out_unique)
{
    uint64_t *d_kin=nullptr,*d_kout=nullptr; uint32_t *d_vin=nullptr,*d_vout=nullptr;
    uint32_t *d_idx_in=nullptr,*d_mult_in=nullptr,*d_sidx=nullptr,*d_smult=nullptr;
    WCK(cudaMalloc(&d_kin,(size_t)n*8)); WCK(cudaMalloc(&d_kout,(size_t)n*8));
    WCK(cudaMalloc(&d_vin,(size_t)n*4)); WCK(cudaMalloc(&d_vout,(size_t)n*4));
    WCK(cudaMalloc(&d_idx_in,(size_t)n*4)); WCK(cudaMalloc(&d_mult_in,(size_t)n*4));
    WCK(cudaMalloc(&d_sidx,(size_t)n*4)); WCK(cudaMalloc(&d_smult,(size_t)n*4));
    WCK(cudaMemcpy(d_kin,h_fps,(size_t)n*8,cudaMemcpyHostToDevice));
    WCK(cudaMemcpy(d_idx_in,h_idx_in,(size_t)n*4,cudaMemcpyHostToDevice));
    WCK(cudaMemcpy(d_mult_in,h_mult_in,(size_t)n*4,cudaMemcpyHostToDevice));
    wms_iota<<<(n+255)/256,256>>>(d_vin,n);

    // 1) sort (fp, j)
    void* t1=nullptr; size_t s1=0;
    cub::DeviceRadixSort::SortPairs(t1,s1,d_kin,d_kout,d_vin,d_vout,n);
    WCK(cudaMalloc(&t1,s1));
    cub::DeviceRadixSort::SortPairs(t1,s1,d_kin,d_kout,d_vin,d_vout,n);
    // gather sorted idx/mult via the permutation d_vout
    wms_gather2<<<(n+255)/256,256>>>(d_idx_in, d_vout, d_sidx, n);
    wms_gather2<<<(n+255)/256,256>>>(d_mult_in, d_vout, d_smult, n);

    // 2) RLE on sorted fps -> run lengths + offsets -> first idx per run
    uint64_t* d_uk=nullptr; uint32_t *d_rl=nullptr,*d_off=nullptr,*d_nr=nullptr,*d_idxout=nullptr;
    WCK(cudaMalloc(&d_uk,(size_t)n*8)); WCK(cudaMalloc(&d_rl,(size_t)n*4));
    WCK(cudaMalloc(&d_off,(size_t)n*4)); WCK(cudaMalloc(&d_nr,4)); WCK(cudaMalloc(&d_idxout,(size_t)n*4));
    void* t2=nullptr; size_t s2=0;
    cub::DeviceRunLengthEncode::Encode(t2,s2,d_kout,d_uk,d_rl,d_nr,n); WCK(cudaMalloc(&t2,s2));
    cub::DeviceRunLengthEncode::Encode(t2,s2,d_kout,d_uk,d_rl,d_nr,n);
    uint32_t nr=0; WCK(cudaMemcpy(&nr,d_nr,4,cudaMemcpyDeviceToHost));
    void* t3=nullptr; size_t s3=0;
    cub::DeviceScan::ExclusiveSum(t3,s3,d_rl,d_off,nr); WCK(cudaMalloc(&t3,s3));
    cub::DeviceScan::ExclusiveSum(t3,s3,d_rl,d_off,nr);
    wms_gather2<<<(nr+255)/256,256>>>(d_sidx, d_off, d_idxout, nr);

    // 3) ReduceByKey(fp, mult, +) -> summed mult per group (same group order as RLE)
    uint64_t* d_uk2=nullptr; uint32_t* d_multout=nullptr; uint32_t* d_nr2=nullptr;
    WCK(cudaMalloc(&d_uk2,(size_t)n*8)); WCK(cudaMalloc(&d_multout,(size_t)n*4)); WCK(cudaMalloc(&d_nr2,4));
    void* t4=nullptr; size_t s4=0;
    cub::DeviceReduce::ReduceByKey(t4,s4,d_kout,d_uk2,d_smult,d_multout,d_nr2,WmsAddU32{},n); WCK(cudaMalloc(&t4,s4));
    cub::DeviceReduce::ReduceByKey(t4,s4,d_kout,d_uk2,d_smult,d_multout,d_nr2,WmsAddU32{},n);

    WCK(cudaMemcpy(h_idx_out,d_idxout,(size_t)nr*4,cudaMemcpyDeviceToHost));
    WCK(cudaMemcpy(h_mult_out,d_multout,(size_t)nr*4,cudaMemcpyDeviceToHost));
    *out_unique=nr;

    cudaFree(d_kin);cudaFree(d_kout);cudaFree(d_vin);cudaFree(d_vout);
    cudaFree(d_idx_in);cudaFree(d_mult_in);cudaFree(d_sidx);cudaFree(d_smult);
    cudaFree(d_uk);cudaFree(d_rl);cudaFree(d_off);cudaFree(d_nr);cudaFree(d_idxout);
    cudaFree(d_uk2);cudaFree(d_multout);cudaFree(d_nr2);
    cudaFree(t1);cudaFree(t2);cudaFree(t3);cudaFree(t4);
    return 0;
}

} // extern "C"
