# Known Issues

## Numerical performance for CUDA Capability 7.x GPUs

All CUDA Capability 7.x GPUs (e.g. V100) produce obviously bad output, with lots
of clashing residues (the clashes cause a ranking score of -99 or lower), unless
the environment variable `XLA_FLAGS` is set to include
`--xla_disable_hlo_passes=custom-kernel-fusion-rewriter`.

## Incorrect handling of two-letter atoms in SMILES ligands

Between commits https://github.com/google-deepmind/alphafold3/commit/f8df1c7 and
https://github.com/google-deepmind/alphafold3/commit/4e4023c, AlphaFold 3
handled incorrectly any two-letter atoms (e.g. Cl, Br) in ligands defined using
SMILES strings.
