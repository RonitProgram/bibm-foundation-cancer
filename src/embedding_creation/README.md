# Embedding creation

bfstuff/ and brnabertstuff/ each contain four per-input scripts.
Default input: data/generated_inputs if that file exists, otherwise data/expression.
Use --input-dir data/expression to force the archived inputs, or --input-dir
data/generated_inputs to require regenerated inputs. --output-dir overrides output.
Defaults write data/generated_embeddings/{bfstuff,brnabertstuff}.

Computational behavior from the original templates is preserved. BulkRNABert
initializes random parameters; it does not load pretrained weights. BulkFormer
uses compatible-key loading and its active configuration is labeled 147M while
the checkpoint filename says 37M. These need separate scientific correction.
Historical templates are saved in provenance/pre_cleanup_sources.zip.
