# AF3 Prediction for 2PV7

This example makes a prediction for 2PV7 which includes: two protein chains, two NAD molecules, and two TYR molecules. 

Included are scripts for running AF3 predictions with:
- `mmseqs` mode: This makes an AF3 prediction using MMseqs (via the ColabFold server) to generate MSAs and templates for all protein chains. This does *not* include MSAs for RNA.
- `singleseq` mode: This makes an AF3 prediction using no MSAs or templates for any chains.

Extending `alphafold_input.json`:
- This example does not include RNA, DNA, or ion prediction. See `example_1ZBI` for an example with all three.
- This example does not include multiple seeds. Just add more, e.g. `[1, 42]`.
- Custom MSAs and templates can be used by explicitly adding `unpairedMsa: af3_preds_mmseqs/mmseqs_env/mmseqs_101.a3m` key-value pair or `unpairedMsa: af3_preds_mmseqs/mmseqs_env/templates_101/` to the protein chain.