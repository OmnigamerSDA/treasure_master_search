# Contributing

This repository is a curated snapshot of an active research project.
The development repo is private and is synced into this one periodically.

## What you can usefully contribute

- **Bug reports** on the public code: forward kernels, password codec,
  validation tools, and documentation.
- **Build fixes** for platforms outside the maintainer's environment
  (non-Linux, non-NVIDIA, alternate CUDA versions, etc.).
- **Performance patches** for the forward CPU, CUDA, and OpenCL paths.
- **Algorithm clarifications** in the `docs/` writeups.

## What is harder to land

- Requests for reverse/CNF/SAT internals — those systems live in the private
  development repo and are not part of this forward-search snapshot.
- Sweeping refactors of `src/common/` — used by every lane.
- New "hypothesis" directions. Active research lives on the dev side.

## Process

1. Open an issue first for anything larger than a one-file fix.
2. PRs against `main`. Keep them focused; one concern per PR.
3. If you touch a forward kernel, please include before/after rates from
   the same machine (a single `--workunit_size` run is enough).

## License and attribution

By contributing you agree your changes are released under this repo's
license. The NES screenshot in the README is third-party; please don't
modify the attribution there.

## Out of scope

This project is **not** a general SAT-solving or reverse-engineering toolkit. It is a focused
attempt at one specific NES code-recovery problem. Issues asking for
generic solver features will usually be closed with a pointer upstream.
