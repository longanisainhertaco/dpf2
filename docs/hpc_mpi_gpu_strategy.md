# MPI + GPU Enablement and Performance Strategy

This document outlines the steps and artifacts required to port the core solvers to a hybrid MPI + GPU implementation, provide profiling-driven validation (including Roofline), ship reproducible containers, and supply restart-capable parallel I/O benchmarks with deployment guidance for HPC systems.

## 1. Porting Core Solvers to MPI + GPU
- **GPU backend:** Target AMReX with CUDA builds; keep HIP/SYCL as future-compatible backends by avoiding vendor-specific intrinsics in solver kernels.
- **Domain decomposition:** Reuse existing MPI decomposition; ensure ghost-zone exchanges use GPU-aware MPI or staged host buffers when GPU-direct is unavailable.
- **Kernel migration:**
  - Identify hot kernels (advance, flux computations, Riemann solves) with the current profiling data.
  - Refactor kernels into AMReX `ParallelFor` calls with device lambdas; encapsulate shared temporary buffers in per-tile scratch space to reduce global memory pressure.
  - Replace manual loops with tiled launch parameters sized from occupancy analysis (`amrex::Gpu::Device::maxThreadsPerBlock`).
- **Memory management:**
  - Allocate long-lived arrays using AMReX-managed allocators (`The_Arena()`) to keep unified ownership across host/device.
  - Use pinned host buffers for boundary exchange packing/unpacking when GPU-direct is disabled; retain async copies to overlap with MPI.
- **Asynchrony:**
  - Structure time-step drivers so that packing, GPU compute, and MPI exchange run in separate CUDA streams with explicit event synchronization.
  - Introduce a progress engine for MPI to advance non-blocking requests while kernels run.
- **Correctness validation:**
  - Compare GPU vs CPU results on deterministic inputs with L1/L2 norms and regression baselines.
  - Ensure bitwise reproducibility for restart files by standardizing reductions and halo-ordering.

## 2. Profiling, Scaling, and Roofline
- **Instrumentation:** Build with `-lineinfo -Xcompiler -fopenmp` for detailed tracebacks; enable AMReX profiling (`amrex.profile=1`) and NVTX ranges around each phase (pack, compute, exchange).
- **Strong/weak scaling runs:**
  - Strong: fix global problem size; vary nodes/GPUs {1,2,4,8,16}. Report time-step wall-clock and solver occupancy.
  - Weak: scale problem size with rank count to keep per-GPU workload constant; collect throughput (zones/sec) and parallel efficiency.
  - Capture runs with and without GPU-direct MPI to quantify PCIe staging overheads.
- **Roofline analysis:**
  - Collect kernel FLOP/s and memory traffic via Nsight Compute (`ncu`) and AMReX statistics.
  - Compute arithmetic intensity and plot against device peak bandwidth/FLOP; flag kernels below the memory roof for tiling/coalescing improvements.
  - Archive profiling traces under `diagnostics/roofline/` with build metadata, input decks, and compiler hashes.

## 3. Containers with Pinned Toolchains
- **Base images:**
  - CPU: `ghcr.io/<org>/solver:cpu-${DATE}` built from Ubuntu 22.04, GCC 12, OpenMPI 4.1, and matching AMReX release.
  - GPU: `ghcr.io/<org>/solver:cuda12-${DATE}` derived from `nvidia/cuda:12.4.0-devel-ubuntu22.04` with CUDA-aware OpenMPI.
- **Pinning and flags:**
  - Record compiler versions and flags (`-O3 -march=native -ffast-math` for CPU; `-O3 -lineinfo --use_fast_math` for CUDA) in `/etc/container_environment` and export `CMAKE_CUDA_ARCHITECTURES` for supported devices.
  - Embed `requirements.lock` and `conan.lock` (if applicable) for deterministic third-party dependencies.
- **Runtime hooks:** Provide `entrypoint` scripts that detect host CUDA driver compatibility, set `AMREX_GPU_VERBOSE`, and warn when GPU-direct is unavailable.

## 4. CI Coverage (CPU + GPU)
- **Correctness smoke tests:**
  - CPU job: small domain (e.g., 32^3) with 2 MPI ranks; verify residual norms and regression output hashes.
  - GPU job: identical deck on a single GPU runner; compare against CPU baseline within tolerance.
- **Performance smoke tests:**
  - Run a 64^3 deck for 5 time steps; assert wall-clock below a threshold and kernel occupancy above a minimum (Nsight Systems summary).
  - Upload profiling summaries as CI artifacts for trend tracking.
- **Build matrix:** Cover `Release` and `RelWithDebInfo`; fail fast on missing CUDA driver or MPI symbols.

## 5. Restart-Capable Parallel I/O Benchmarks
- **Benchmark harness:**
  - Add a driver that writes checkpoint and plotfiles using AMReX `VisMF` with `async_io` enabled; include options for stride, aggregation size, and compression.
  - Support restart validation by reading the checkpoint and resuming for N steps, comparing diagnostics to a fresh run.
- **Metrics:** Capture bandwidth (GB/s), metadata overhead, open-file counts, and restart correctness hashes. Provide scaling sweeps over file counts and node counts.
- **Artifacts:** Store benchmark inputs under `benchmarks/restart_io/` with templates for Lustre and GPFS (stripe counts, block sizes). Include parsers for job scheduler outputs.

## 6. HPC Deployment Guidance
- **Schedulers:** Provide SLURM/LSF templates with optimal binding flags (`--gpu-bind`, `--mca btl`) and environment hints for GPU-direct.
- **Filesystem tuning:** Recommend Lustre stripe counts, GPFS block sizes, and collective buffering settings based on benchmark results.
- **Monitoring:** Suggest integration with node-level telemetry (DCGM, pcm) and log scrapers to track throughput and error rates.

## 7. Deliverables Checklist
- GPU-enabled solver binaries validated against CPU baselines.
- Strong/weak scaling reports with plots and Roofline figures under `docs/perf/`.
- Container images published to GHCR with pinned dependencies and documented flags.
- CI matrix covering CPU/GPU correctness and performance smoke tests.
- Restart-capable parallel I/O benchmarks with scheduler templates and restart verification.
