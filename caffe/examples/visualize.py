from sklearn.manifold import TSNE
import pickle

model = TSNE(n_components=2, random_state=0)
vec = pickle.load(open('feat_train.pickle', 'rb'))
model.fit_transform(vec)

import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
