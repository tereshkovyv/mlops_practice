from pathlib import Path

import numpy as np


def load_mnist_via_sklearn():
	from sklearn.datasets import fetch_openml
	from sklearn.model_selection import train_test_split

	X, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False)
	X = X.reshape(-1, 28, 28).astype(np.uint8)
	y = y.astype(np.int64)
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
	return (x_train, y_train), (x_test, y_test)


def save_dataset(images, labels, out_dir: Path):
	out_dir.mkdir(parents=True, exist_ok=True)
	path = out_dir / 'data.npz'
	np.savez_compressed(path, images=images, labels=labels)


def main():
	print('---started---')
	(x_train, y_train), (x_test, y_test) = load_mnist_via_sklearn()
	save_dataset(x_train, y_train, Path('data/train'))
	save_dataset(x_test, y_test, Path('data/test'))
	print('---created_dataset---')

main()

