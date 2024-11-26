# AF3 Prediction for 1ZBI

This example makes a prediction for 1ZBI which includes: two protein chains, one RNA chain, one DNA chain, and four Magnesium ions. 

Included are scripts for running AF3 predictions:
- `run_af3_mmseqs.sh`: This makes an AF3 prediction using MMseqs (via the ColabFold server) to generate MSAs and templates for all protein chains. This does *not* include MSAs for RNA.
- `run_af3_singleseq.sh`: This makes an AF3 prediction using no MSAs or templates for any chains.
- `run_af3_custom.sh`: This makes an AF3 prediction using custom MSAs and templates contained in the `custom_inputs` directory. See the differences between `alphafold_input.json` and `alphafold_input_custom.json` to see how these custom inputs are specified.

Extending this:
- This example does not include a small molecule prediction. See `example_2PV7` for an example.
- This example does not include residue modifications for the protein, RNA, or DNA chains. See `example_2PV7` for an example of residue modifications for proteins.
