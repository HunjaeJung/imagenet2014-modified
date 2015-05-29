from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
import seaborn as sns
import util

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


def scatter(x, colors):
    colors = np.array(colors)

    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 20))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(20):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def main():
    n = 1000
    (X_train, y_train), (X_test, y_test) = util.load_feat_vec()
    digits_proj = TSNE(random_state=130).fit_transform(X_train[:n])
    del X_train
    del X_test
    del y_test

    scatter(digits_proj, y_train[:n])
    plt.savefig('t-sne_embedding_{}.png'.format(n), dpi=120)

if __name__ == '__main__':
    main()
