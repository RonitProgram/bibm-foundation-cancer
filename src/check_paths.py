"""Audit local BRCA dependencies without loading models or full arrays."""
from pathlib import Path
import ast
import importlib.util
import json
import math
import struct

ROOT = Path(__file__).resolve().parents[1]

def array_info(path):
    if not path.is_file():
        return {'path': str(path), 'status': 'missing'}
    with path.open('rb') as f:
        if f.read(6) != b'\x93NUMPY':
            return {'path': str(path), 'status': 'invalid_npy_magic'}
        version = f.read(2)
        n = 2 if version[0] == 1 else 4
        length = struct.unpack('<H' if n == 2 else '<I', f.read(n))[0]
        header = ast.literal_eval(f.read(length).decode().strip())
        expected = f.tell() + math.prod(header['shape']) * int(header['descr'][2:])
    actual = path.stat().st_size
    return {'path': str(path), 'shape': header['shape'], 'actual_bytes': actual,
            'expected_bytes': expected, 'status': 'complete' if actual == expected else 'incomplete'}

def main():
    syntax = []
    for path in (ROOT / 'src').rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
        syntax.append(str(path.relative_to(ROOT)))
    inputs = ['X.csv', 'yiv.csv', 'X_hvg_1000.csv', 'X_hvg_10000.csv',
              'X_litgenes.csv', 'X_litgenes2.csv', 'gProfiler_hsapiens_2026-07-30_17-08-56.csv']
    assets = ['BulkFormer_37M.pt', 'G_tcga.pt', 'G_tcga_weight.pt',
              'esm2_feature_concat (1).pt', 'bulkformer_gene_info (1).csv']
    required = [ROOT/'data/expression'/n for n in inputs]
    required += [ROOT/'data/model_assets'/n for n in assets]
    required += [ROOT/'data/literature/litgenes2_ensembl_ids.txt',
                 ROOT/'third_party/multiomics-open-research/data/bulkrnabert/common_gene_id.txt',
                 ROOT/'notebooks/classification.ipynb',
                 ROOT/'results/tables/cv_metrics_long_by_datatype_model.csv']
    arrays = []
    for folder,names in {
        'bfstuff':['hvg1000.npy','hvg10000.npy','litgenesbrca.npy','litgenesbrca2.npy'],
        'bertstuff':['hvg1000bert.npy','hvg10000bert.npy','litgenesbert.npy','litgenesbert2.npy']
    }.items():
        arrays += [array_info(ROOT/'data/embeddings'/folder/n) for n in names]
    external = [array_info(ROOT.parent/'breaststuff'/n) for n in ['hvg1000.npy','hvg10000.npy']]
    report = {
        'python_sources_parsed': syntax,
        'missing_required_small_inputs': [str(p) for p in required if not p.is_file()],
        'archived_arrays': arrays,
        'external_arrays_not_copied': external,
        'packages_available': {n:importlib.util.find_spec(n) is not None
                               for n in ['numpy','pandas','matplotlib','sklearn','torch','jax','haiku']},
        'legacy_hpc_paths_in_active_python': [str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.py')
                                              if p.name != 'check_paths.py' and '/lustre/' in p.read_text(encoding='utf-8')],
        'models_or_classifiers_executed': False
    }
    output=ROOT/'provenance/path_audit.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Parsed {len(syntax)} Python sources; missing small inputs: {len(report["missing_required_small_inputs"])}')
    print('Archived arrays:', ', '.join(Path(a['path']).name+': '+a['status'] for a in arrays))
    print(f'Legacy HPC paths: {report["legacy_hpc_paths_in_active_python"]}')
    print(f'Full report: {output}')

if __name__ == '__main__':
    main()
