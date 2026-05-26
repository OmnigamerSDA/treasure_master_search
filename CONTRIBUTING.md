# Contributing

This repository is a curated snapshot of an active research project.
The development repo is private and is synced into this one periodically.

## What you can usefully contribute

- **Bug reports** on the public code: forward kernels, reverse engines,
  CNF generator, password codec, validation tools.
- **Build fixes** for platforms outside the maintainer's environment
  (non-Linux, non-NVIDIA, alternate CUDA versions, etc.).
- **Performance patches** for the forward CPU SIMD and CUDA paths.
- **Algorithm clarifications** in the `docs/` and `wiki/` writeups.

## What is harder to land

- Changes to the CNF emission shape — those are coupled to the
  in-development solver-binding pipeline that lives in the private repo
  and may not be visible here.
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

This project is **not** a general SAT-solving toolkit. It is a focused
attempt at one specific NES code-recovery problem. Issues asking for
generic solver features will usually be closed with a pointer upstream.
