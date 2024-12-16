# AF3 Prediction for Multiple Inputs

This example makes a prediction for both 1ZBI and 2PV7. (See the other examples for single predictions of these.)

Included are scripts for running AF3 predictions:
- `run_af3_mmseqs.sh`: This makes an AF3 prediction using MMseqs (via the ColabFold server) to generate MSAs and templates for all protein chains. This does *not* include MSAs for RNA.
- `run_af3_singleseq.sh`: This makes an AF3 prediction using no MSAs or templates for any chains.
- `run_af3_custom.sh`: This makes an AF3 prediction using custom MSAs and templates contained in the `custom_inputs` directory.
