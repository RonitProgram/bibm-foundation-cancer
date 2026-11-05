"""RECONSTRUCTED VARIANT: existing template with dataset and local paths changed.
Not evidence of the exact historical generator of saved result arrays.
Model initialization, normalization, masking, and inference logic are preserved.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
import argparse
import sys
sys.path.insert(0, str(ROOT / 'src'))
from pipeline_paths import expression_input
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-dir', type=Path, help='Override input directory; default prefers generated inputs over archived inputs')
parser.add_argument('--output-dir', type=Path, help='Override embedding output directory')
args = parser.parse_args()
OUTPUT_DIR = ROOT / "data/generated_embeddings/brnabertstuff"
if args.output_dir is not None:
    OUTPUT_DIR = args.output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# The template initializes RANDOM parameters with PRNGKey(0).
# This variant does not introduce pretrained checkpoint loading.
import os 
import numpy as np 
import pandas as pd 
# import scipy.stats as stats
# from scipy.stats import pearsonr 
# import matplotlib.pyplot as plt 
# import seaborn as sns 
import tarfile

basic_dir = str(ROOT / "data/expression")
X = pd.read_csv(expression_input('X_hvg_1000.csv', args.input_dir), index_col=0)
X.index.name = "patient_id"


X.columns = X.columns.str.split(".").str[0]

print("Before:", X.shape)

X = X.loc[:, ~X.columns.duplicated()]

print("After:", X.shape)
print("Duplicate columns:", X.columns.duplicated().sum())

import pandas as pd

common_genes = pd.read_csv(
    str(ROOT / "third_party/multiomics-open-research/data/bulkrnabert/common_gene_id.txt"),
    header=None
)[0].tolist()


# The original X columns are your selected HVGs.
hvg_genes = X.columns.tolist()

# Embedding axis follows the common_genes order because X_full was reindexed.
common_gene_to_idx = {
    gene: idx
    for idx, gene in enumerate(common_genes)
}

# Preserve the original HVG ranking/order.
selected_hvg_genes = [
    gene
    for gene in hvg_genes
    if gene in common_gene_to_idx
]

missing_hvg_genes = [
    gene
    for gene in hvg_genes
    if gene not in common_gene_to_idx
]

interested_gene_idx = np.asarray(
    [
        common_gene_to_idx[gene]
        for gene in selected_hvg_genes
    ],
    dtype=np.int64,
)

print("HVGs after removing duplicates:", len(hvg_genes))
print("HVGs found in BulkRNABert:", len(selected_hvg_genes))
print("HVGs missing from BulkRNABert:", len(missing_hvg_genes))
print("Selected index count:", len(interested_gene_idx))

assert len(interested_gene_idx) == len(selected_hvg_genes)
assert len(np.unique(interested_gene_idx)) == len(interested_gene_idx)
assert interested_gene_idx.min() >= 0
assert interested_gene_idx.max() < len(common_genes)


X_bulkbert = X.loc[:, X.columns.intersection(common_genes)]

X_bulkbert = X_bulkbert.reindex(
    columns=common_genes,
    fill_value=0
)

print("X shape:", X.shape)
print("X_bulkbert shape:", X_bulkbert.shape)

import sys
sys.path.insert(0, str(ROOT / "third_party/multiomics-open-research"))

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from multiomics_open_research.bulk_rna_bert.config import BulkRNABertConfig
from multiomics_open_research.bulk_rna_bert.model import build_bulk_rna_bert_forward_fn
from multiomics_open_research.common.preprocess import preprocess_omic
from multiomics_open_research.common.tokenizer import BinnedOmicTokenizer

config = BulkRNABertConfig(
    n_genes=len(common_genes),
    n_expressions_bins=64,
    embed_dim=64,
    init_gene_embed_dim=64,
    project_gene_embedding=False,
    use_gene_embedding=False,
    num_attention_heads=4,
    key_size=16,
    ffn_embed_dim=128,
    num_layers=2,
    use_memory_efficient_attention=False,
    use_gradient_checkpointing=False,
    gene2vec_weights_path="data/gene2vec_weights_common_gene_ids.npy",
    embeddings_layers_to_save=(2,),
    attention_layers_to_save=(),
    use_log_normalization=True,
    use_max_normalization=True,
    normalization_factor=5.547176906585117,
)

forward_fn = build_bulk_rna_bert_forward_fn(
    model_config=config,
    compute_dtype=jnp.float32,
    param_dtype=jnp.float32,
    output_dtype=jnp.float32,
    model_name="bulk_bert",
)

tokenizer = BinnedOmicTokenizer(
    n_expressions_bins=config.n_expressions_bins,
    use_max_normalization=config.use_max_normalization,
    normalization_factor=config.normalization_factor,
    prepend_cls_token=False,
)

rng = jax.random.PRNGKey(0)
params = None
transformed_forward = hk.transform(forward_fn)
params = transformed_forward.init(rng, jnp.zeros((1, 64), dtype=jnp.int32))


# ...existing code...
import numpy as np
from tqdm import tqdm

# Use full aligned matrix (already reindexed to common_genes)
X_full = X_bulkbert

# def encode_full_dataset(X_df, batch_size=4):
#     all_embeddings = []
#     for i in tqdm(range(0, len(X_df), batch_size)):
#         xb = X_df.iloc[i:i+batch_size]
#         gene_expression = preprocess_omic(xb, config)
#         print("After preprocess:", gene_expression.shape)
#         tokens = tokenizer.batch_tokenize(gene_expression)
#         print("Tokens shape:", tokens.shape)

#         outputs = transformed_forward.apply(params, None, tokens)
#         print("Forward pass finished")
#         emb = np.asarray(outputs["embeddings_2"], dtype=np.float32)  # (B, G, D)
#         patient_embedding = emb.mean(axis=1)   # (B, 64)
#         all_embeddings.append(patient_embedding)

#     return np.concatenate(all_embeddings, axis=0)

import numpy as np
from tqdm import tqdm


def encode_full_dataset(
    X_df,
    interested_gene_idx,
    output_path,
    batch_size=4,
):
    interested_gene_idx = np.asarray(
        interested_gene_idx,
        dtype=np.int64,
    )

    output_array = None
    output_row = 0

    for i in tqdm(range(0, len(X_df), batch_size)):
        xb = X_df.iloc[i:i + batch_size]

        gene_expression = preprocess_omic(xb, config)
        tokens = tokenizer.batch_tokenize(gene_expression)

        outputs = transformed_forward.apply(
            params,
            None,
            tokens,
        )

        # Shape: (batch, all_genes, embedding_dimension)
        embeddings = np.asarray(
            outputs["embeddings_2"],
            dtype=np.float32,
        )

        if interested_gene_idx.max() >= embeddings.shape[1]:
            raise IndexError(
                f"Gene index {interested_gene_idx.max()} exceeds "
                f"the model output's {embeddings.shape[1]} genes"
            )

        # Keep only the requested genes—do not average them.
        selected_embeddings = embeddings[
            :,
            interested_gene_idx,
            :,
        ]

        if output_array is None:
            output_shape = (
                len(X_df),
                len(interested_gene_idx),
                selected_embeddings.shape[2],
            )

            output_array = np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype=np.float32,
                shape=output_shape,
            )

            print(
                f"Saving embeddings with shape {output_shape}",
                flush=True,
            )

        next_row = output_row + selected_embeddings.shape[0]

        output_array[output_row:next_row] = selected_embeddings
        output_row = next_row

    output_array.flush()

    print(f"Saved embeddings to {output_path}", flush=True)
    return output_path

output_path = str(OUTPUT_DIR / 'hvg1000bert.npy')
encode_full_dataset(X_full, interested_gene_idx, output_path, batch_size=4)
print(f"Saved embeddings to {output_path}")