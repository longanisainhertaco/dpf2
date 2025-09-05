import json
import os
import time
from datetime import datetime
from pathlib import Path
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

def _matmul_chunk(args):
    a_chunk, b = args
    return a_chunk @ b

def strong_scaling():
    N = 400
    a = np.random.rand(N, N)
    b = np.random.rand(N, N)
    results = []
    for p in [1, 2, 4]:
        t0 = time.time()
        if p == 1:
            c = a @ b
        else:
            chunks = np.array_split(a, p)
            with mp.Pool(p) as pool:
                pieces = pool.map(_matmul_chunk, [(chunk, b) for chunk in chunks])
            c = np.vstack(pieces)
        dt = time.time() - t0
        flops = 2 * N ** 3
        intensity = flops / (3 * N ** 2 * 8)
        gflops = flops / dt / 1e9
        results.append({"p": p, "time": dt, "gflops": gflops, "intensity": intensity})
    return results

def weak_scaling():
    N0 = 200
    results = []
    for p in [1, 2, 4]:
        N = int(N0 * np.sqrt(p))
        a = np.random.rand(N, N)
        b = np.random.rand(N, N)
        t0 = time.time()
        if p == 1:
            c = a @ b
        else:
            chunks = np.array_split(a, p)
            with mp.Pool(p) as pool:
                pieces = pool.map(_matmul_chunk, [(chunk, b) for chunk in chunks])
            c = np.vstack(pieces)
        dt = time.time() - t0
        flops = 2 * N ** 3
        intensity = flops / (3 * N ** 2 * 8)
        gflops = flops / dt / 1e9
        results.append({"p": p, "N": N, "time": dt, "gflops": gflops, "intensity": intensity})
    return results

def io_benchmark():
    sizes = [10**6, 5*10**6, 10**7]
    results = []
    for size in sizes:
        data = np.random.rand(size)
        t0 = time.time(); np.save('tmp.npy', data); write_t = time.time() - t0
        t0 = time.time(); _ = np.load('tmp.npy'); read_t = time.time() - t0
        os.remove('tmp.npy')
        bytes_ = size * 8
        results.append({"size": size, "write_MBps": bytes_ / write_t / 1e6, "read_MBps": bytes_ / read_t / 1e6})
    return results

def plot_scaling(strong, weak, out_dir):
    p = [r["p"] for r in strong]
    t = [r["time"] for r in strong]
    plt.figure(); plt.plot(p, t, marker='o'); plt.xlabel('Processes'); plt.ylabel('Time (s)'); plt.title('Strong Scaling');
    plt.savefig(out_dir / 'strong_scaling.png'); plt.close()

    p = [r["p"] for r in weak]
    t = [r["time"] for r in weak]
    plt.figure(); plt.plot(p, t, marker='o'); plt.xlabel('Processes'); plt.ylabel('Time (s)'); plt.title('Weak Scaling');
    plt.savefig(out_dir / 'weak_scaling.png'); plt.close()

def plot_io(io, out_dir):
    sizes = [r["size"] / 1e6 for r in io]
    w = [r["write_MBps"] for r in io]
    r = [r["read_MBps"] for r in io]
    plt.figure(); plt.plot(sizes, w, marker='o', label='write'); plt.plot(sizes, r, marker='s', label='read');
    plt.xlabel('Size (MB)'); plt.ylabel('Throughput (MB/s)'); plt.title('I/O Benchmark'); plt.legend();
    plt.savefig(out_dir / 'io_benchmark.png'); plt.close()

def plot_roofline(strong, out_dir):
    peak_flops = 200 # GFLOPS
    peak_bw = 50     # GB/s
    intensity = np.logspace(-2, 3, 100)
    roof = np.minimum(peak_flops, peak_bw * intensity)
    plt.figure()
    plt.loglog(intensity, roof, label='Roofline')
    pts_int = [r['intensity'] for r in strong]
    pts_perf = [r['gflops'] for r in strong]
    plt.scatter(pts_int, pts_perf, color='red', label='measured')
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
    plt.ylabel('Performance (GFLOP/s)')
    plt.legend()
    plt.savefig(out_dir / 'roofline.png')
    plt.close()

def main():
    out_dir = Path('docs/performance')
    out_dir.mkdir(parents=True, exist_ok=True)
    strong = strong_scaling()
    weak = weak_scaling()
    io = io_benchmark()
    plot_scaling(strong, weak, out_dir)
    plot_io(io, out_dir)
    plot_roofline(strong, out_dir)
    manifest = {
        "container_hash": "2076a760c691e081427db6496e5804944dd6f2c1cae7ca55906a95a0ba85beb6",
        "compiler_flags": os.environ.get('CFLAGS', ''),
        "generated": datetime.utcnow().isoformat()
    }
    with open(out_dir / 'run_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print('Benchmark complete')

if __name__ == '__main__':
    main()
