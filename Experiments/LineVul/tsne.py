import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(rc={'figure.figsize': (11.7, 8.27)})
palette = sns.color_palette("bright", 2)


def plot_embedding(X_org, y, title=None, new=True):
    # X, _, Y, _ = train_test_split(X_org, y, test_size=0.5)
    # X, Y = np.asarray(X), np.asarray(Y)
    # X = X[:10000]
    # Y = Y[:10000]
    # y_v = ['Vulnerable' if yi == 1 else 'Non-Vulnerable' for yi in Y]
    if not new and os.path.exists(str(title) + '-tsne-features.json'):
        file = open(str(title) + '-tsne-features.json', 'r')
        _x, _y = json.load(file)
        X = np.array(_x)
        Y = np.array(_y)
    else:
        tsne = manifold.TSNE(n_components=2, init='pca',
                             random_state=0, n_jobs=-2)
        print('Fitting TSNE!')
        X = tsne.fit_transform(X)
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        file_ = open(str(title) + '-tsne-features.json', 'w')
        if isinstance(X, np.ndarray):
            _x = X.tolist()
            _y = Y.tolist()
        else:
            _x = X
            _y = Y
        json.dump([_x, _y], file_)
        file_.close()
    sns.set(style='white')
    plt.figure(title, figsize=(10, 10), edgecolor='black')
    plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1],
                marker='.', c="darkgrey", s=12, linewidth=3.5)
    plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1],
                marker='.', c="black", s=12, linewidth=3.5)
    plt.xticks([]), plt.yticks([])
    # sns.scatterplot(X[:, 0], X[:, 1], hue=y_v, palette=['red', 'green'])
    # for i in range(X.shape[0]):
    #     if Y[i] == 0:
    #         plt.scatter(X[i, 0], X[i, 1], marker='.', c="darkgrey", s=10)
    #     else:
    #         plt.scatter(X[i, 0], X[i, 1], marker='.', c="black", s=10)
    # plt.scatter()
    # plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title("")
    plt.tight_layout()
    plt.savefig(title + '.jpeg', dpi=1000)
    plt.show()


if __name__ == '__main__':
    x_a = np.random.uniform(0, 1, size=(32, 256))
    targets = np.random.randint(0, 2, size=(32))
    print(targets)
    plot_embedding(x_a, targets)
    print("Computing t-SNE embedding")
