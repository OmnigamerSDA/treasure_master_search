#!/usr/bin/env bash
# raceway_autoconfig.sh — CPU PRODUCTION-engine auto-configure.
#
# The bounded-wave raceway is the production CPU forward engine (2026-06-16) — best across
# BOTH throughput and memory; the non-dedup/screen/state-dedup harnesses are research only.
# This script detects the host's ISA/compiler/topology and selects the FASTEST available
# raceway build (it only ever builds raceway kernels), plus a sensible run configuration.
#
# Build preference (fastest raceway kernel first; see cpu_raceway.cpp header + operating guide):
#   cpu_raceway        AVX-512 natmap  (x10/blmerge) — PRODUCTION primary, needs AVX-512 + clang>=19
#   cpu_raceway_u      AVX-512 universal (g++, x12)  — fallback when clang>=19 is absent
#   cpu_raceway_avx2   AVX2 natmap (x2/blmerge)      — older-ISA hosts without AVX-512
#
# This script probes /proc/cpuinfo for AVX-512, finds the best available compiler
# (clang>=19 preferred for natmap; g++ otherwise), reads the LLC size to size the
# bounded wave, and reads the SMT sibling layout to suggest physical/SMT pin sets.
#
# Usage:
#   ./raceway_autoconfig.sh            # detect + print plan (no build)
#   ./raceway_autoconfig.sh --build    # also build the selected target
#   ./raceway_autoconfig.sh --build --run 0x9e9d137b   # build, then a sample run
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

BUILD=0; RUNKEY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) BUILD=1; shift ;;
    --run)   RUNKEY="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

say(){ echo "[raceway-autoconfig] $*"; }

# ---- ISA -------------------------------------------------------------------
HAVE_AVX512=0
grep -qw avx512f /proc/cpuinfo 2>/dev/null && HAVE_AVX512=1
HAVE_AVX2=0
grep -qw avx2 /proc/cpuinfo 2>/dev/null && HAVE_AVX2=1

# ---- compiler --------------------------------------------------------------
# natmap wants clang>=19; otherwise g++. Probe newest clang first.
CLANG=""
for c in clang++-21 clang++-20 clang++-19 clang++; do
  if command -v "$c" >/dev/null 2>&1; then
    maj=$("$c" -dumpversion 2>/dev/null | cut -d. -f1)
    if [[ -n "$maj" && "$maj" -ge 19 ]]; then CLANG="$c"; break; fi
  fi
done
HAVE_GXX=0; command -v g++ >/dev/null 2>&1 && HAVE_GXX=1

# ---- target decision -------------------------------------------------------
# AVX-512 host + clang>=19 -> natmap (primary). AVX-512 but no clang -> universal (g++).
# No AVX-512 but AVX2 -> avx2 legacy build (clang preferred, g++ ok).
TARGET=""; CXX=""; MAKEARGS=()
if (( HAVE_AVX512 )); then
  if [[ -n "$CLANG" ]]; then
    TARGET="cpu_raceway"; CXX="$CLANG"; MAKEARGS=(CXX="$CLANG" cpu_raceway)
  elif (( HAVE_GXX )); then
    TARGET="cpu_raceway_u"; CXX="g++"; MAKEARGS=(cpu_raceway_u)
  fi
elif (( HAVE_AVX2 )); then
  if [[ -n "$CLANG" ]]; then CXX="$CLANG"; else CXX="g++"; fi
  TARGET="cpu_raceway_avx2"; MAKEARGS=(CXX="$CXX" cpu_raceway_avx2)
fi
if [[ -z "$TARGET" ]]; then
  say "ERROR: no usable target — need at least AVX2 + a C++20 compiler (clang++ or g++)."
  exit 1
fi

# ---- topology: physical cores + SMT siblings -> pin sets -------------------
NPROC=$(nproc 2>/dev/null || echo 1)
# Pick the LOWEST logical CPU per physical core (visit CPUs in NUMERIC order so the
# representative set is the natural 0..NPHYS-1 on the common N/N+NPHYS sibling layout).
declare -A seen_core=()
PHYS=()
for cpu in $(seq 0 $((NPROC-1))); do
  d=/sys/devices/system/cpu/cpu$cpu
  [[ -d "$d" ]] || continue
  cid=$(cat "$d/topology/core_id" 2>/dev/null || echo "$cpu")
  pid=$(cat "$d/topology/physical_package_id" 2>/dev/null || echo 0)
  key="$pid:$cid"
  if [[ -z "${seen_core[$key]:-}" ]]; then seen_core[$key]=1; PHYS+=("$cpu"); fi
done
NPHYS=${#PHYS[@]}; (( NPHYS == 0 )) && NPHYS=$NPROC
PHYS_PIN=$(IFS=,; echo "${PHYS[*]}")
SMT_PIN="0-$((NPROC-1))"

# ---- LLC size -> wave_N ----------------------------------------------------
# wave uses 2 buffers of wave_N*128 B. Target each buffer ~ <= half the LLC so the
# live wave is cache-resident (§4). Default 131072 (=32 MiB/buf) is good for >=32 MiB L3.
LLC_KB=0
for idx in 3 2; do
  f=$(ls /sys/devices/system/cpu/cpu0/cache/index*/level 2>/dev/null | while read l; do
        lv=$(cat "$l"); [[ "$lv" == "$idx" ]] && dirname "$l"; done | head -1)
  if [[ -n "$f" ]]; then
    sz=$(cat "$f/size" 2>/dev/null)         # e.g. "32768K" or "96M"
    case "$sz" in
      *K) LLC_KB=${sz%K} ;;
      *M) LLC_KB=$(( ${sz%M} * 1024 )) ;;
    esac
    [[ "$LLC_KB" != 0 ]] && break
  fi
done
WAVE=131072
if (( LLC_KB > 0 )); then
  # half the LLC, in 128-B records, rounded down to a power of two-ish (just use the value)
  half_records=$(( (LLC_KB * 1024 / 2) / 128 ))
  (( half_records > 0 )) && WAVE=$half_records
  # clamp to a sane range
  (( WAVE < 16384 )) && WAVE=16384
  (( WAVE > 262144 )) && WAVE=262144
fi

# ---- report ----------------------------------------------------------------
echo
say "ISA            : AVX-512=$HAVE_AVX512  AVX2=$HAVE_AVX2"
say "compiler       : clang>=19='${CLANG:-none}'  g++=$HAVE_GXX  -> using $CXX"
say "topology       : $NPHYS physical core(s), $NPROC logical (SMT $(( NPROC>NPHYS ? 1 : 0 )))"
say "selected raceway build (PRODUCTION engine): $TARGET"
say "build command  : make ${MAKEARGS[*]}"
say "LLC            : ${LLC_KB} KiB  -> wave_N=$WAVE"
say "pin (SMT off)  : taskset -c $PHYS_PIN   (T=$NPHYS)"
say "pin (SMT on)   : taskset -c $SMT_PIN    (T=$NPROC)"
echo
SAMPLE="taskset -c $SMT_PIN ./$TARGET <key> 0 $NPROC 5 count $WAVE 22 4"
say "sample full-key run (full 2^32 needs the producer cap):"
say "  PRODUCER_CAP=1 PCAP_BITS=24 $SAMPLE"

# ---- build / run -----------------------------------------------------------
if (( BUILD )); then
  echo; say "building: make ${MAKEARGS[*]}"
  make "${MAKEARGS[@]}" || { say "build FAILED"; exit 3; }
  say "built ./$TARGET"
  if [[ -n "$RUNKEY" ]]; then
    echo; say "sample run (W16M, T=$NPROC):"
    set -x
    RACEWAY_CAP=1 taskset -c "$SMT_PIN" "./$TARGET" "$RUNKEY" 16777216 "$NPROC" 5 count "$WAVE" 22 4
  fi
fi
