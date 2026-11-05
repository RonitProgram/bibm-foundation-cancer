"""Shared local paths. No model training or third-party imports."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def expression_input(filename, input_dir=None):
    if input_dir is not None:
        path = Path(input_dir) / filename
    else:
        generated = ROOT / 'data/generated_inputs' / filename
        path = generated if generated.is_file() else ROOT / 'data/expression' / filename
    if not path.is_file():
        raise FileNotFoundError(f'Missing expression input: {path}')
    return path

def embedding_input(model, filename, source='archived'):
    folder = {'bf': 'bfstuff', 'bert': 'bertstuff'}[model]
    if source == 'generated':
        folder = 'bfstuff' if model == 'bf' else 'brnabertstuff'
        return ROOT / 'data/generated_embeddings' / folder / filename
    return ROOT / 'data/embeddings' / folder / filename
