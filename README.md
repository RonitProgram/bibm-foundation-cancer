# BRCA reproducibility

BRCA-only code, archived data and results traced from the breast figure in blresults.
Original project files outside this folder are unchanged. Reconstructed scripts
are distinguished from historical evidence in provenance/.

## Layout

- src/gene_input_creation: four input-generation scripts.
- src/embedding_creation/bfstuff and brnabertstuff: four embedding scripts per model.
- src/embedding_evaluation: HVG1k/HVG10k and both curated-set evaluations.
- src/results and src/plotting: report parsing and exact archived-figure reproduction.
- data/expression: archived expression matrices and labels.
- data/embeddings/bfstuff and bertstuff: complete archived embeddings available locally.
- data/model_assets, literature, selection: model assets and input-selection evidence.
- results/raw_reports and tables: archived reports and figure metric tables.
- figures/original: original BRCA figure and key.
- notebooks/classification.ipynb: expression baseline code AND saved outputs used by parser.
- third_party: model code and license notices.
- provenance: audits, historical references and pre-cleanup source snapshot.

## Pipeline and paths

1. Run a script in src/gene_input_creation; outputs go to data/generated_inputs.
2. Run its model script in src/embedding_creation. It prefers a generated input if
   present and otherwise uses the archived input. Use --input-dir to choose explicitly.
   Outputs go to data/generated_embeddings/bfstuff or brnabertstuff.
3. Evaluation defaults to archived arrays. Use --embeddings-source generated for
   step 2 outputs. --top-k selects 156/312/0 (uncapped) in HVG evaluation;
   --literature-set selects 1/2 in curated evaluation. New reports go to
   results/generated_reports and never overwrite historical reports by default.
4. To rebuild historical tables from saved reports/notebook outputs, run
   python src/results/build_grouped_results.py. Outputs go to results/rebuilt.
   This builder reads CV means, not held-out test scores. It intentionally defaults
   to historical reports; use --brcaresults-dir explicitly for another report folder.
5. To reproduce the archived figure, run:

```powershell
python -m pip install -r requirements-plotting.txt
python src/plotting/reproduce_brca_figure.py
```

The plotter reads the archived results/tables and writes figures/regenerated.
Both regenerated images previously matched the originals pixel-for-pixel.
Run python src/check_paths.py for a dependency/array-integrity report.

## Known limits

- Saved labels/matrices contain 1,051 samples (1,032 zero, 19 positive), not 1,052.
- BulkFormer HVG arrays are missing from this collection. The external HVG1k
  transfer is now complete; external HVG10k remains incomplete at latest audit.
  No large files were added during cleanup. See provenance/path_audit.json.
- The figure shows CV means. Available HVG code selects flattened embedding
  coordinates once on the outer training set before classifier CV, not genes
  within each validation fold. These CV scores therefore reuse feature-selection
  information across the inner validation folds.
- Historical baseline notebook state is required to parse its saved results;
  the notebook is not a verified clean-kernel end-to-end pipeline. It reads a
  selection-count vector different from the filename written in an earlier cell,
  and evaluates the best-recall fold model on held-out samples.
- BulkRNABert templates initialize random weights. BulkFormer architecture/config
  and checkpoint labels differ, with partial compatible-key loading. The organized
  pipeline does not correct or validate these model choices.
- Original historical HVG/156/uncapped generator versions were not recovered;
  new scripts/parameter options are reconstructions, not proof of provenance.
- The second literature membership comes from the saved CSV header; the original
  spreadsheet-to-Ensembl mapping procedure was not recovered.
- Bare arrays cannot establish sample identity/order. No model inference or
  classifier training was run during cleanup. Current Python lacks scikit-learn;
  full evaluation requires a suitable ML/CUDA environment. Dependency lists are
  import references, not a recovered version-locked environment.

The cleanup removes redundant active templates, duplicate figure copies and a
duplicate vocabulary table; historical sources are preserved in a compact ZIP.
No expression matrices, complete archived embeddings, reports or model weights
were deleted. data/ is excluded from normal Git commits by .gitignore.
