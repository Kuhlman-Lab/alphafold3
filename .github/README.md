# Kuhlman Lab Installation of AlphaFold3

## Getting Started
**Note:** AF3 weights **must be acquired and used** as outlined by the [official AF3 repository](https://github.com/google-deepmind/alphafold3).

### Installation
1. Make a new conda/mamba environment:
`mamba create -p .conda/envs/af3 python=3.11`

2. Activate the new environment:
`mamba activate af3`

3. Clone this GitHub repository:
`https://github.com/Kuhlman-Lab/alphafold3.git`

3. Install AF3 Python dependencies:
`pip3 install -r dev-requirements.txt`

4. Install the AF3 source code (need GNU g++ >= 9.4):
`pip3 install --no-deps .`

5. Run `build_data` (this was created in step 4):
`build_data`
