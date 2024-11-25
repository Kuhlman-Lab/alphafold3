# Kuhlman Lab Installation of AlphaFold3

This is an unofficial repo that wraps around [Google Deepmind's AlphaFold3](https://github.com/google-deepmind/alphafold3). It extends some of the official repo's functionality and utilizes [ColabFold](https://github.com/sokrypton/ColabFold) and its MMseqs2 server for protein MSA and template generation.

## Getting Started

### Installation
1. Make a new conda/mamba environment:
`mamba create -p .conda/envs/af3 python=3.11`

2. Activate the new environment:
`mamba activate af3`

3. Clone this GitHub repository:
`https://github.com/Kuhlman-Lab/alphafold3.git`

4. Install extra dependencies:
`mamba install zlib gcc_linux-64 gxx_linux-64 requests`

5. Install CUDA Toolkit 12.6:
`mamba install -c nvidia cuda-toolkit=12.6`

6. Install AF3 Python dependencies:
`pip3 install -r dev-requirements.txt`

7. Install the AF3 source code:
`pip3 install --no-deps .`

8. Run `build_data` (this was created in step 7):
`build_data`

### AF3 Weights
AF3 weights **must be acquired and used** as outlined by the [official AF3 repository](https://github.com/google-deepmind/alphafold3).

Once acquired, make a new directory called `models` in the base `alphafold3` folder and place the weights file (e.g. `af3.bin.zst`) inside.