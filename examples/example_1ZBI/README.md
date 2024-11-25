# AF3 Prediction for 1ZBI

This example makes a prediction for 1ZBI which includes: two protein chains, one RNA chain, one DNA chain, and four Magnesium ions. 

Included are scripts for running AF3 predictions with:
- `mmseqs` mode: This makes an AF3 prediction using MMseqs (via the ColabFold server) to generate MSAs and templates for all protein chains. This does *not* include MSAs for RNA.
- `singleseq` mode: This makes an AF3 prediction using no MSAs or templates for any chains.

Extending this:
- This example does not include a small molecule prediction. See `example_2PV7` for an example.
- This example does not include residue modifications for the protein, RNA, or DNA chains. See `example_2PV7` for an example of residue modifications for proteins.
- This example does not include multiple seeds. Just add more, e.g. `[1, 42]`.
- Custom MSAs and templates can be used by explicitly adding `unpairedMsa: af3_preds_mmseqs/mmseqs_env/mmseqs_101.a3m` key-value pair or `unpairedMsa: af3_preds_mmseqs/mmseqs_env/templates_101/` to the protein chain.