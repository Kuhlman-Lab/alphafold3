# AF3 Prediction for 2PV7

This example makes a prediction for 2PV7 which includes: two protein chains, two NAD molecules, and two TYR molecules. 

Included are scripts for running AF3 predictions with:
- `run_af3_mmseqs.sh`: This makes an AF3 prediction using MMseqs (via the ColabFold server) to generate MSAs and templates for all protein chains. This does *not* include MSAs for RNA.
- `run_af3_singleseq.sh`: This makes an AF3 prediction using no MSAs or templates for any chains.
- `run_af3_custom.sh`: This makes an AF3 prediction using custom MSAs and templates contained in the `custom_inputs` directory. See the differences between `alphafold_input.json` and `alphafold_input_custom.json` to see how these custom inputs are specified.

Extending `alphafold_input.json`:
- This example does not include RNA, DNA, or ion prediction. See `example_1ZBI` for an example with all three.
- This example does not include residue modifications for the RNA or DNA chains.