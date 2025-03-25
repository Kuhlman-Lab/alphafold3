# Kuhlman Lab Installation of AlphaFold3

This is an unofficial repo that wraps around [Google Deepmind's AlphaFold3](https://github.com/google-deepmind/alphafold3). It extends some of the official repo's functionality and utilizes [ColabFold](https://github.com/sokrypton/ColabFold) and its MMseqs2 server for protein MSA and template generation.

This repository will be updated frequently (with both official and unofficial changes) so be sure to pull the newest version from GitHub often. **If files in `src/alphafold3` change, you MUST re-run Step 7 from Installation below to reinstall AF3.**

Last commit merged from official AF3 repo: `69b749c`

## Getting started

### Installation
1. Make a new conda/mamba environment:
`mamba create -p .conda/envs/af3 python=3.11`

2. Activate the new environment:
`mamba activate af3`

3. Clone this GitHub repository:
`https://github.com/Kuhlman-Lab/alphafold3.git`

4. Install extra dependencies:
`mamba install zlib gcc_linux-64 gxx_linux-64 requests hmmer -c bioconda`

5. Install CUDA Toolkit 12.6:
`mamba install -c nvidia cuda-toolkit=12.6`

6. Install AF3 Python dependencies:
`pip3 install -r dev-requirements.txt`

7. Install the AF3 source code:
`pip3 install --no-deps .`

8. Run `build_data` (this was created in step 7):
`build_data`

### AF3 weights
AF3 weights **must be acquired and used** as outlined by the [official AF3 repository](https://github.com/google-deepmind/alphafold3).

Once acquired, make a new directory called `models` in the base `alphafold3` folder and place the weights file (e.g. `af3.bin.zst`) inside.

### First prediction

After installation and getting access to the AF3 weights, you're ready for your first prediction! 

For this, we'll create a JSON file specifying our input. Create a new file called `alphafold_input.json` with the contents:
```
{
    "name": "Top7",
    "sequences": [
      {
        "protein": {
          "id": ["A"],
          "sequence": "MGDIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQLEGGSLE"
        }
      }
    ],
    "modelSeeds": [1]
}
```
This file specifies that we want to make an AF3 prediction with 1 seed for Top7, which consists of a single protein chain. 

Now, we'll simply call the `run_af3.py` script, point to this file, use `--output_dir af3_preds` to contain our predictions, and include `--run_mmseqs` to indicate we want to use MMseqs to generate MSAs and templates for our query:
`python run/run_af3.py --json_path alphafold_input.json --output_dir af3_preds --run_mmseqs`

Once you see 
```
Done processing fold input Top7.
Done processing 1 fold inputs.
```
your prediction has completed! You'll find the predicted structures (`*.cif`) and some confidence predictions in your new `af3_preds` folder.

For more complicated inputs and examples, please refer to the [input format documentation](../docs/input.md) and the `examples` folder.


## Have suggestions or want to contribute?

If you have any suggestions about features you want to see enabled, please open a new issue and let us know! We're also actively welcoming contributions; just create a fork and submit a PR to this repo. We'll get back to you shortly and hopefully merge your commits! 
