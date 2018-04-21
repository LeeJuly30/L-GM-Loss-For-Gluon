import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patheffects as PathEffects
import mxnet as mx
from mxnet import nd


def plot_features(features, label):
    vis_feature = features
    vis_label = label
    unique_label = np.unique(vis_label)
    for i, _label in enumerate(unique_label):
        vis_label[vis_label==_label] = i
    name_dict = dict()
    for i, _label in enumerate(unique_label):
        name_dict[i] = str(int(_label))
    f = plt.figure(figsize=(14, 14))
    palette = np.array(sns.color_palette("hls", unique_label.shape[0]))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(vis_feature[:, 0], vis_feature[:, 1], lw=0, s=40,
                        c=palette[vis_label.astype(np.int)])
    ax.axis('tight')
    txts = []
    for i, _label in enumerate(unique_label):
        xtext, ytext = np.median(vis_feature[vis_label == i, :], axis=0)
        txt = ax.text(xtext, ytext, name_dict[i])
    plt.show()


def evaluate_accuracy(data_iterator, net, l_gm, ctx):
    acc = mx.metric.Accuracy()

    features, predicts, labels = [], [], []
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        
        fts = net(data)
        loss, probability = l_gm(fts, label)

        predictions = nd.argmax(probability, axis=1)
        acc.update(preds=predictions, labels=label)

        features.extend(fts.asnumpy())
        predicts.extend(predictions.asnumpy())
        labels.extend(label.asnumpy())

    features = np.array(features)
    predicts = np.array(predicts)
    labels = np.array(labels)

    return acc.get()[1], features, predicts, labels