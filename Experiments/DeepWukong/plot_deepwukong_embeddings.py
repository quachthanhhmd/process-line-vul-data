import pickle 
from random import sample
    
import random 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json

sns.set(rc={'figure.figsize':(6,6)})
sns.set_context("notebook", rc={"lines.linewidth": 4.5})
sns.set_style("whitegrid", {"axes.facecolor": "none"})
sns.set_palette("Greys")
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.edgecolor'] = 'black'

def plot_embedding(embeddings, y):
    X, Y = np.asarray(embeddings), np.asarray(y)   
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X = tsne.fit_transform(X)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    if isinstance(X, np.ndarray):
        _x = X.tolist()
        _y = Y.tolist()
    else:
        _x = X
        _y = Y
    return _x, _y
   
def calculate_centroids(_features, _labels):
    pos = []
    neg = []
    for f, l  in zip(_features, _labels):
        if l == 1:
            pos.append(f)
        else:
            neg.append(f)
    posx = [x[0] for x in pos]
    posy = [x[1] for x in pos]
    negx = [x[0] for x in neg]
    negy = [x[1] for x in neg]
    _px = np.median(posx)
    _py = np.median(posy)
    _nx = np.median(negx)
    _ny = np.median(negy)
    return (_px, _py), (_nx, _ny)


def calculate_distance(p1, p2):
    return np.abs(np.sqrt(((p1[0] - p2[0])*(p1[0] - p2[0])) + ((p1[1] - p2[1])*(p1[1] - p2[1]))))


parser = argparse.ArgumentParser()
parser.add_argument("--hidden_states_path", type=str, required=True,
                    help="path to hidden_states_test.pickle")
args = parser.parse_args()
hidden_states_path = args.hidden_states_path


with open(hidden_states_path,"rb") as f:
    xfg_data = pickle.load(f)

x,y=xfg_data[0],xfg_data[1]


_features, _labels = plot_embedding(x,y)


features = []
labels = []
r = 1.0
for f, l in zip(_features, _labels):
    if f[0] <= r and f[1] <= r:
        features.append([f[0] * (1/r), f[1]* (1/r)])
        labels.append(l)
                   
pmed, nmed = calculate_centroids(features, labels)
dist = calculate_distance(pmed, nmed)
X = np.array(features)
Y = np.array(labels)
x0,y0,x1,y1=[],[],[],[]
for i,j in enumerate(Y):
    if j==0:
        x0.append(X[i][0])
        y0.append(X[i][1])
    else:
        x1.append(X[i][0])
        y1.append(X[i][1])


plt0 = plt.scatter(x0, y0, marker='.', c="darkgrey", s=10)
plt1 = plt.scatter(x1, y1, marker='.',c="black",s=10)
plt.xticks([]), plt.yticks([])
plt.show()