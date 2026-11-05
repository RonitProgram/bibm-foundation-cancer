"""Reconstructed from hvg.ipynb; not a recovered historical script.
Rank genes on the 80% training partition; preserve all sample rows in output.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd

def select_hvgs_bulk(
    X,
    n_top_genes,
    log_transform=True,
    exclude_technical=True
):
    """
    Select highly variable genes from patient-level expression data.

    Parameters
    ----------
    X : pandas.DataFrame
        Rows = patients, columns = genes.
        Values should be normalized expression, TPM/CPM, or counts.
    n_top_genes : int
        Number of genes to retain.
    log_transform : bool
        Apply log2(x + 1). Set False if data are already log-transformed.
    """

    X = X.copy()

    if log_transform:
        X = np.log2(X + 1)

    if exclude_technical:
        technical = X.columns.str.match(
            r"^(MT-|RPL|RPS|MTRNR|IGH|IGK|IGL)"
        )
        X = X.loc[:, ~technical]

    # Median absolute deviation is more robust than ordinary variance
    gene_medians = X.median(axis=0)
    mad = X.sub(gene_medians, axis=1).abs().median(axis=0)

    hvg_names = mad.nlargest(n_top_genes).index.tolist()

    return hvg_names, X[hvg_names]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT/'data/expression/X.csv')
    parser.add_argument('--labels', type=Path, default=ROOT/'data/expression/yiv.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'data/generated_inputs/X_hvg_10000.csv')
    args = parser.parse_args()
    X = pd.read_csv(args.input, index_col=0)
    y = pd.read_csv(args.labels).iloc[:,0].to_numpy()
    if len(X) != len(y):
        raise ValueError('Expression and label row counts differ')
    train, _ = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    names, _ = select_hvgs_bulk(X.iloc[train], n_top_genes=10000)
    result = np.log2(X.loc[:, names] + 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output)
    print(f'Saved {result.shape} to {args.output}')

if __name__ == '__main__':
    main()
