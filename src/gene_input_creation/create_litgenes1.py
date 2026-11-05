"""Reconstructed curated expression-input generator; preserves sample order."""
from pathlib import Path
import argparse
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT/'data/expression/X.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'data/generated_inputs/X_litgenes.csv')
    parser.add_argument('--mapping', type=Path, default=ROOT/'data/expression/gProfiler_hsapiens_2026-07-30_17-08-56.csv')
    args = parser.parse_args()
    X = pd.read_csv(args.input, index_col=0)
    X.columns = X.columns.str.split('.').str[0]
    X = X.loc[:, ~X.columns.duplicated()]
    genes = pd.read_csv(args.mapping, index_col=0)['converted_alias'].dropna()
    result = X.loc[:, X.columns.isin(genes)]
    if result.shape[1] == 0:
        raise ValueError('No requested genes matched the expression matrix')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output)
    print(f'Saved {result.shape} to {args.output}')

if __name__ == '__main__':
    main()
