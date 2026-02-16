import pickle
from pathlib import Path

import numpy as np

from sklearn.neural_network import MLPClassifier


def load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with np.load(path) as data:
        images = data['images']
        labels = data['labels']
    return images, labels


def preprocess(images):
    if images.ndim == 3:
        n, h, w = images.shape
        images = images.reshape(n, h * w)
    images = images.astype(np.float32) / 255.0
    return images


def main():
    print('---start_training___')
    data_path = Path('data/train/data.npz')
    images, labels = load_npz(data_path)
    X = preprocess(images)
    y = labels

    clf = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=200, random_state=42)
    clf.fit(X, y)

    model_dir = Path('model')
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path('model/model.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(clf, f)

    print('---end_training___')

main()
