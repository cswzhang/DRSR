# coding=utf-8
import logging

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)


def _classification(embedding, labels_np, split_ratio=0.7):
    labels_np = shuffle(labels_np)
    nodes = labels_np[:, 0]
    labels = labels_np[:, 1]

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    train_size = int(labels_np.shape[0] * split_ratio)
    features = embedding[nodes]

    train_x = features[:train_size, :]
    train_y = labels[:train_size, :]
    test_x = features[train_size:, :]
    test_y = labels[train_size:, :]
    clf = OneVsRestClassifier(
        LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=3000))

    clf.fit(train_x, train_y)
    y_pred = clf.predict_proba(test_x)
    y_pred = lb.transform(np.argmax(y_pred, 1))
    acc = np.sum(np.argmax(y_pred, 1) == np.argmax(test_y, 1)) / len(y_pred)
    eval_dict = {
        'acc': acc,
        'f1-micro': metrics.f1_score(np.argmax(test_y, 1), np.argmax(y_pred, 1),
                                     average='micro'),
        'f1-macro': metrics.f1_score(np.argmax(test_y, 1), np.argmax(y_pred, 1),
                                     average='macro'),
    }
    return list(eval_dict.values())


def classification(embedding, labels_np, split_ratio=0.7, loop=100):
    eval_dict = {
        'acc': 0.0,
        'f1-micro': 0.0,
        'f1-macro': 0.0,
        'std-micro': 0.0,
        'std-macro': 0.0
    }
    temp = []
    for _ in range(loop):
        temp.append(_classification(embedding, labels_np, split_ratio))
    temp = np.array(temp)
    eval_dict['acc'] = np.round(np.mean(temp[:, 0]), 4)
    eval_dict['f1-micro'] = np.round(np.mean(temp[:, 1]), 4)
    eval_dict['f1-macro'] = np.round(np.mean(temp[:, 2]), 4)
    eval_dict['std-micro'] = np.round(np.std(temp[:, 1]), 4)
    eval_dict['std-macro'] = np.round(np.std(temp[:, 2]), 4)
    print(eval_dict)
    return eval_dict


def _clustering(embedding, labels_np):
    labels_np = shuffle(labels_np)
    nodes = labels_np[:, 0]
    labels = labels_np[:, 1]
    features = embedding[nodes]

    model = KMeans(n_clusters=max(labels) + 1)
    # model = BayesianGaussianMixture(n_components=max(labels) + 1)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    model.fit(features)

    # c_pred = model.labels_
    c_pred = model.predict(features)

    nmi = metrics.normalized_mutual_info_score(np.argmax(labels, 1), c_pred)

    eval_dict = {
        'homogeneity': metrics.homogeneity_score(np.argmax(labels, 1), c_pred),
        'completeness': metrics.completeness_score(np.argmax(labels, 1), c_pred),
        'silhouette': metrics.silhouette_score(features, c_pred),
        'NMI': nmi
    }
    # print(eval_dict)
    return eval_dict


def clustering(embedding, labels_np, loop=10):
    eval_dict = {
        'homogeneity': 0.0,
        'completeness': 0.0,
        'silhouette': 0.0,
        'NMI': 0
    }
    for _ in range(loop):
        tmp_dict = _clustering(embedding, labels_np)
        for key in tmp_dict.keys():
            eval_dict[key] += tmp_dict[key]
    for key in tmp_dict.keys():
        eval_dict[key] = round((1.0 * eval_dict[key]) / loop, 4)
    print(eval_dict)
    return eval_dict
