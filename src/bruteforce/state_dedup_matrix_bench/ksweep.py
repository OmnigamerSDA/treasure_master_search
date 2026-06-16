import subprocess, statistics
B = "./state_dedup_matrix_bench"

def run(impl, thr, k, reps=5, win=65536, keys="/tmp/keys128.txt"):
    vals = []
    for _ in range(reps):
        subprocess.run([B, "--keys", keys, "--out", "/tmp/r.csv", "--windows", str(win),
            "--impl", impl, "--interleave12", "--repeats", "5", "--flat-only", "--scaling",
            "--threads", str(thr), "--label", "x",
            "--dedup-every-maps", str(k), "--first-dedup-maps", "1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        vals.append(float(open("/tmp/r.csv").read().splitlines()[1].split(",")[5]))
    return statistics.median(vals), min(vals), max(vals)

names = {"tm_avx512_r512_map_8": "nat", "tm_avx512_r512s_8": "uni"}
print("=== K sweep @ 16t, W65536, 128 keys, median-of-5 (M/s) ===")
print("%-4s %-4s %8s %8s %8s" % ("impl", "K", "median", "min", "max"))
best = {}
for impl in ("tm_avx512_r512_map_8", "tm_avx512_r512s_8"):
    for k in (1, 2, 3, 4, 6, 8):
        m, lo, hi = run(impl, 16, k)
        print("%-4s %-4d %8.3f %8.3f %8.3f" % (names[impl], k, m, lo, hi))
        best.setdefault(impl, (0, 0))
        if m > best[impl][1]:
            best[impl] = (k, m)
    print()
print("best K:", {names[i]: best[i] for i in best})
