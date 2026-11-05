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
OUTPUT_DIR = ROOT / "data/generated_embeddings/bfstuff"
if args.output_dir is not None:
    OUTPUT_DIR = args.output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# %%
import os

# %%
import sys
sys.path.insert(0, str(ROOT / "third_party/BulkFormer"))

# %%
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader,random_split
from torch_geometric.typing import SparseTensor
from utils.BulkFormer import BulkFormer
from model.config import model_params

# %%
# Configuration
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU is visible. Submit this script through the GPU Slurm job.")
device = torch.device("cuda")
print(f"Using device: {device} ({torch.cuda.get_device_name(0)})", flush=True)
graph_path = str(ROOT / "data/model_assets" / 'G_tcga.pt')
weights_path = str(ROOT / "data/model_assets" / 'G_tcga_weight.pt')
gene_emb_path = str(ROOT / "data/model_assets" / 'esm2_feature_concat (1).pt')

# %%
# Initialize the BulkFormer model with preloaded graph structure and gene embeddings.
graph = torch.load(graph_path, map_location='cpu', weights_only=False)
weights = torch.load(weights_path, map_location='cpu', weights_only=False)
graph = SparseTensor(row=graph[1], col=graph[0], value=weights).t().to(device)
gene_emb = torch.load(gene_emb_path, map_location='cpu', weights_only=False)
model_params['graph'] = graph
model_params['gene_emb'] = gene_emb
model = BulkFormer(**model_params).to(device)

# %%
# Load the pretrained BulkFormer model checkpoint for inference or fine-tuning.
ckpt_model = torch.load(str(ROOT / "data/model_assets" / 'BulkFormer_37M.pt'), map_location='cpu', weights_only=False)

new_state_dict = OrderedDict()
for key, value in ckpt_model.items():
    new_key = key[7:] if key.startswith("module.") else key
    new_state_dict[new_key] = value

# Safe checkpoint loading: only keep keys that exist in current model and match shape
model_state = model.state_dict()
compatible_state = OrderedDict()
skipped = []

for k, v in new_state_dict.items():
    if k in model_state and model_state[k].shape == v.shape:
        compatible_state[k] = v
    else:
        skipped.append(k)

load_info = model.load_state_dict(compatible_state, strict=False)

print(f"Loaded keys: {len(compatible_state)}")
print(f"Skipped keys (missing or shape mismatch): {len(skipped)}")
print(f"Still missing in model after load: {len(load_info.missing_keys)}")
print(f"Unexpected keys after filtering: {len(load_info.unexpected_keys)}")

# %%
def normalize_data(X_df, gene_length_dict):
    """
    Normalize RNA-seq count data to log-transformed TPM values.

    Parameters
    ----------
    X_df : pandas.DataFrame
        A gene expression matrix where rows represent samples and columns represent genes.
        Each entry contains the raw read count of a gene in a given sample.

    gene_length_dict : dict
        A dictionary mapping gene identifiers (Ensembl gene IDs) to gene lengths (in base pairs).

    Returns
    -------
    log_tpm_df : pandas.DataFrame
        A DataFrame of the same shape as `X_df`, containing log-transformed TPM values
        (i.e., log(TPM + 1)) for each gene in each sample.

    Description
    -----------
    This function converts raw RNA-seq count data to transcripts per million (TPM) values by
    normalizing for gene length and sample-specific total expression. Gene lengths are provided
    via `gene_length_dict`, and genes not present in the dictionary are assigned a default
    length of 1,000 bp (equivalent to no correction). The resulting TPM values are subsequently
    log-transformed using the natural logarithm (log1p). This normalization procedure accounts
    for both gene length and sequencing depth, facilitating cross-sample and cross-gene comparisons.
    """
    gene_names = X_df.columns
    gene_lengths_kb = np.array([gene_length_dict.get(gene, 1000) / 1000  for gene in gene_names])
    counts_matirx = X_df.values
    rate = counts_matirx / gene_lengths_kb
    sum_per_sample = rate.sum(axis=1)
    sum_per_sample[sum_per_sample == 0] = 1e-6  
    sum_per_sample = sum_per_sample.reshape(-1, 1)
    tpm = rate / sum_per_sample * 1e6
    log_tpm = np.log1p(tpm)
    log_tpm_df = pd.DataFrame(log_tpm,index=X_df.index, columns=X_df.columns)
    return log_tpm_df

# %%
def main_gene_selection(X_df, gene_list):
    """
    Aligns a gene expression matrix to a predefined gene list by adding placeholder values
    for missing genes and generating a binary mask indicating imputed entries.

    Parameters
    ----------
    X_df : pandas.DataFrame
        A gene expression matrix with rows representing samples and columns representing genes.
        The entries are typically log-transformed or normalized expression values.

    gene_list : list of str
        A predefined list of gene identifiers (Ensembl Gene IDs) to be retained
        in the final matrix. This list defines the desired gene space for downstream analyses.

    Returns
    -------
    X_df : pandas.DataFrame
        A gene expression matrix aligned to `gene_list`, with missing genes filled with a constant
        placeholder value (−10) and columns ordered accordingly.

    to_fill_columns : list of str
        A list of genes from `gene_list` that were not present in the original `X_df`
        and were therefore added with placeholder values.

    var : pandas.DataFrame
        A DataFrame with one row per gene, containing a binary column `'mask'` indicating
        whether a gene was imputed (1) or originally present (0). This can be used for masking
        in training or evaluation of models that distinguish observed and imputed entries.

    Notes
    -----
    This function ensures that all samples share a consistent gene space, which is essential
    for tasks such as model training, cross-dataset integration, or visualization. Placeholder
    values (−10) are used to maintain matrix shape while avoiding unintended bias in downstream
    statistical analyses or machine learning models.
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))

    padding_df = pd.DataFrame(np.full((X_df.shape[0], len(to_fill_columns)), -10), 
                            columns=to_fill_columns, 
                            index=X_df.index)

    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns,var

# %%
def extract_feature(expr_array, 
                    output_feature_type,
                    aggregate_type,
                    device,
                    batch_size,
                    mask_prob = 0.1,
                    interested_gene_idx = None,
                    output_expr = False,
                    esm2_emb = None):
    
    """
    Extracts sample-level or gene-level feature representations from input expression profiles
    using a pre-trained deep learning model.
    """

    device = torch.device(device)
    use_cuda = device.type == "cuda"

    # DataLoader pinning only works for CPU tensors. Keep the dataset on CPU
    # and transfer one pinned batch at a time to the allocated GPU.
    expr_tensor = torch.as_tensor(expr_array, dtype=torch.float32, device="cpu")

    # Create dataset and dataloader for batched inference
    mydataset = TensorDataset(expr_tensor)
    myloader = DataLoader(
        mydataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=0,
    )

    # Switch model to evaluation mode
    model.to(device)
    model.eval()

    all_emb_list = []
    all_pred_expr_list = []

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_cuda):

        if output_feature_type == 'sample_level':
            for (X,) in tqdm(myloader, total=len(myloader), desc="Sample embeddings", mininterval=5):
                X = X.to(device, non_blocking=use_cuda)

                gene_emb = model(X, mask_prob=mask_prob, output_expr=output_expr)
                gene_emb = gene_emb.detach().cpu().numpy()

                # Select interested genes if user provided gene indices
                if interested_gene_idx is not None:
                    if isinstance(interested_gene_idx, torch.Tensor):
                        selected_idx = interested_gene_idx.detach().cpu().numpy()
                    else:
                        selected_idx = np.asarray(interested_gene_idx)
                    gene_emb = gene_emb[:, selected_idx, :]

                if aggregate_type == 'max':
                    final_emb = np.max(gene_emb, axis=1)
                elif aggregate_type == 'mean':
                    final_emb = np.mean(gene_emb, axis=1)
                elif aggregate_type == 'median':
                    final_emb = np.median(gene_emb, axis=1)
                elif aggregate_type == 'all':
                    max_emb = np.max(gene_emb, axis=1)
                    mean_emb = np.mean(gene_emb, axis=1)
                    median_emb = np.median(gene_emb, axis=1)
                    final_emb = max_emb + mean_emb + median_emb
                else:
                    raise ValueError(f"Unknown aggregate_type: {aggregate_type}")

                all_emb_list.append(final_emb)

                # Avoid per-batch cache flush; it adds overhead and can slow inference
                del gene_emb

            result_emb = np.vstack(all_emb_list)
            # result_emb = torch.tensor(result_emb, device='cpu', dtype=torch.float32)

        # elif output_feature_type == 'gene_level':
        #     for (X,) in tqdm(myloader, total=len(myloader), desc="Gene embeddings", mininterval=5):
        #         X = X.to(device, non_blocking=use_cuda)

        #         gene_emb = model(X, mask_prob=mask_prob, output_expr=output_expr)
        #         gene_emb = gene_emb.detach().cpu().numpy()
        #         all_emb_list.append(gene_emb)

        #         del gene_emb

        #     all_emb = np.vstack(all_emb_list)
        #     all_emb_tensor = torch.from_numpy(all_emb).to(dtype=torch.float32)
        #     esm2_emb_cpu = torch.as_tensor(esm2_emb).detach().cpu().to(dtype=torch.float32)
        #     if interested_gene_idx is not None:
        #         esm2_emb_cpu = esm2_emb_cpu[selected_idx]
        #     esm2_emb_expanded = esm2_emb_cpu.unsqueeze(0).expand(
        #         all_emb_tensor.shape[0], -1, -1)
        #     result_emb = torch.cat([all_emb_tensor, esm2_emb_expanded], dim=-1)
        elif output_feature_type == "gene_level":
            if interested_gene_idx is None:
                raise ValueError(
                    "interested_gene_idx is required for gene-level output"
                )

            if isinstance(interested_gene_idx, torch.Tensor):
                selected_idx = (
                    interested_gene_idx.detach().cpu().numpy()
                )
            else:
                selected_idx = np.asarray(
                    interested_gene_idx,
                    dtype=np.int64,
                )

            for (X,) in tqdm(
                myloader,
                total=len(myloader),
                desc="Gene embeddings",
                mininterval=5,
            ):
                X = X.to(device, non_blocking=use_cuda)

                gene_emb = model(
                    X,
                    mask_prob=mask_prob,
                    output_expr=False,
                )

                # Select HVGs before transferring output to CPU.
                gene_emb = gene_emb[:, selected_idx, :]
                all_emb_list.append(
                    gene_emb.detach().cpu().numpy()
                )

                del gene_emb

            all_emb_tensor = torch.from_numpy(
                np.concatenate(all_emb_list, axis=0)
            ).float()

            esm2_emb_cpu = (
                torch.as_tensor(esm2_emb)
                .detach()
                .cpu()
                .float()
            )

            # Select exactly the same genes and preserve the same order.
            esm2_emb_cpu = esm2_emb_cpu[selected_idx]

            esm2_emb_expanded = esm2_emb_cpu.unsqueeze(0).expand(
                all_emb_tensor.shape[0],
                -1,
                -1,
            )

            result_emb = torch.cat(
                [all_emb_tensor, esm2_emb_expanded],
                dim=-1,
            )
        elif output_feature_type == 'expr_level':
            output_expr = True
            for (X,) in tqdm(myloader, total=len(myloader), desc="Expression predictions", mininterval=5):
                X = X.to(device, non_blocking=use_cuda)

                pred_expr = model(X, mask_prob=mask_prob, output_expr=output_expr)
                all_pred_expr_list.append(pred_expr.detach().cpu().numpy())
        else:
            raise ValueError(f"Unknown output_feature_type: {output_feature_type}")

    if output_expr: 
        return np.vstack(all_pred_expr_list)
    return result_emb

# %%
# Load demo count data (raw count)
basic_dir = str(ROOT / "data/expression")
X = pd.read_csv(expression_input('X_litgenes.csv', args.input_dir), index_col=0)
X.index.name = "patient_id"
# y = pd.read_csv(os.path.join(basic_dir, "y.csv"))

# %%
X.columns = X.columns.str.split(".").str[0]

# %%
print("Before:", X.shape)

X = X.loc[:, ~X.columns.duplicated()]

print("After:", X.shape)
print("Duplicate columns:", X.columns.duplicated().sum())

# %%
log_tpm_df = X.apply(np.log1p)

# %%
gene_length_df = pd.read_csv(str(ROOT / "data/model_assets" / 'bulkformer_gene_info (1).csv'))

# robustly find a usable length column (avoids KeyError: 'length')
length_col_candidates = ['length', 'gene_length', 'len', 'Length', 'gene_len', 'transcript_length']
length_col = next((c for c in length_col_candidates if c in gene_length_df.columns), None)

if length_col is None:
	gene_length_dict = {g: 1000 for g in gene_length_df['ensg_id'].tolist()}
else:
	gene_length_dict = gene_length_df.set_index('ensg_id')[length_col].to_dict()

# %%
bulkformer_gene_info = pd.read_csv(str(ROOT / "data/model_assets" / 'bulkformer_gene_info (1).csv'))
bulkformer_gene_list = bulkformer_gene_info['ensg_id'].to_list()

# %%
# Align expression data to a predefined gene list with placeholder imputation for missing genes.
input_df, to_fill_columns, var= main_gene_selection(X_df=log_tpm_df,gene_list=bulkformer_gene_list)

# %%
# Compute the gene-missing rate (masking probability) for the input data.
# This measures the proportion of genes in the sample that are absent from the pretrained vocabulary of 200,10 genes. 
mask_prob = len(list(var[var['mask'] == 1].index)) / var.shape[0]
print(f"mask prob: {mask_prob}")

# %%
# Load the list of genes of interest (each entry must be an Ensembl gene ID
# and must exist in the BulkFormer vocabulary). These genes will be used to
# aggregate their corresponding embeddings into a sample-level representation.
# X contains the requested HVGs after removing versions and duplicates.
hvg_genes = X.columns.tolist()

bulkformer_gene_to_idx = {
    gene: idx for idx, gene in enumerate(bulkformer_gene_list)
}

selected_genes = [
    gene for gene in hvg_genes
    if gene in bulkformer_gene_to_idx
]

missing_hvgs = [
    gene for gene in hvg_genes
    if gene not in bulkformer_gene_to_idx
]

interested_gene_idx = np.asarray(
    [bulkformer_gene_to_idx[gene] for gene in selected_genes],
    dtype=np.int64,
)

print(f"HVGs in input: {len(hvg_genes)}")
print(f"HVGs found in BulkFormer: {len(selected_genes)}")
print(f"HVGs missing from BulkFormer: {len(missing_hvgs)}")

assert len(interested_gene_idx) <= 10000

# %%
# Extract sample-level embedding
res1 = extract_feature(
    expr_array=input_df.values,
    interested_gene_idx=interested_gene_idx,
    output_feature_type='gene_level',
    aggregate_type='mean',
    device=device,
    batch_size=4 if device.type == 'cuda' else 2,
    output_expr=False,
    mask_prob=mask_prob,
    esm2_emb=model_params['gene_emb'])
print(res1.shape)

# %%
np.save(OUTPUT_DIR / 'litgenesbrca.npy', res1.cpu().numpy() if isinstance(res1, torch.Tensor) else res1)
print(f"Saved {OUTPUT_DIR / 'litgenesbrca.npy'}")

