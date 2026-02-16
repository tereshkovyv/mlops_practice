import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score


def load_npz(path: Path):
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
    print('---start_testing---')
    data_path = Path('data/test/data.npz')
    model_path = Path('model/model.pkl')

    images, labels = load_npz(data_path)
    X = preprocess(images)
    y = labels

    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f'Accuracy: {acc:.4f}')

main()
