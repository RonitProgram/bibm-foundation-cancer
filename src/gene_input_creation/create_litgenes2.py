"""Reconstructed curated expression-input generator; preserves sample order."""
from pathlib import Path
import argparse
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT/'data/expression/X.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'data/generated_inputs/X_litgenes2.csv')
    parser.add_argument('--gene-list', type=Path, default=ROOT/'data/literature/litgenes2_ensembl_ids.txt')
    args = parser.parse_args()
    X = pd.read_csv(args.input, index_col=0)
    X.columns = X.columns.str.split('.').str[0]
    X = X.loc[:, ~X.columns.duplicated()]
    # Recovered membership from saved X_litgenes2.csv, not a new spreadsheet mapping.
    genes = args.gene_list.read_text().splitlines()
    result = X.loc[:, X.columns.isin(genes)]
    if result.shape[1] == 0:
        raise ValueError('No requested genes matched the expression matrix')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output)
    print(f'Saved {result.shape} to {args.output}')

if __name__ == '__main__':
    main()
