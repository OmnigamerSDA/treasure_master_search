# src/bruteforce/Makefile (public) — builds the CPU receiver used by the cert-tier operator.
#
# inspect_bonus2_survivors is the CPU verifier that the raceway clearing path forwards its
# other-world / dual checksum survivors to: it re-forwards each candidate on the CPU and gates it
# on valid machine code (the shared shape predicates in ../common/other_world_shape.h), recording
# only all-entries-valid candidates as alerts. cert_tier_ops.py --engine raceway requires it.
#
# The forward CPU engines (SIMD kernels, state-dedup, cpu_raceway) each build from their own
# subdirectory Makefile; this top-level file only builds the receiver.
CC     = g++
CFLAGS = -I../common -I. -Icpu -std=c++11 -O2 -m64

all: inspect_bonus2_survivors

inspect_bonus2_survivors: inspect_bonus2_survivors.cpp \
		../common/key_schedule.cpp ../common/rng_obj.cpp ../common/alignment2.cpp \
		../common/tm_base.cpp cpu/tm_8.cpp
	$(CC) $(CFLAGS) \
		../common/key_schedule.cpp ../common/rng_obj.cpp ../common/alignment2.cpp ../common/tm_base.cpp \
		cpu/tm_8.cpp inspect_bonus2_survivors.cpp -o inspect_bonus2_survivors

clean:
	rm -f inspect_bonus2_survivors
