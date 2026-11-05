# Embedding evaluation

The HVG scripts accept --top-k 156, 312 (default), or 0 (uncapped selection).
The literature script accepts --literature-set 1 or 2 (default).
All three accept --embeddings-source archived (default) or generated, and --output.
Archived arrays are resolved from data/embeddings/{bfstuff,bertstuff}; generated
arrays from data/generated_embeddings/{bfstuff,brnabertstuff}. Outputs default to
results/generated_reports/{archived,generated}, preserving historical reports.

Selection/scoring algorithms are preserved. Reconstructed parameter variants do
not prove exact reproduction of historical reports. See the root README.
