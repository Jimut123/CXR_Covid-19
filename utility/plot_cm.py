EPOCHS = 50
MODEL_NAME = 'InceptionV3'
IMG_SIZE = '360x360'
OUTPUT_LAYERS = '128-32-3'



index = {'normal': 0, 'covid': 1,  'pneumonia': 2}
rev_index = {0: 'normal',1: 'covid', 2: 'pneumonia'}


import pickle

with open('y_test_list_50e_InceptionV3_360x360_128-32-3.pkl', 'rb') as f:
    y_test_list = pickle.load(f)

with open('y_pred_list_50e_InceptionV3_360x360_128-32-3.pkl', 'rb') as f:
    y_pred_list = pickle.load(f)
# y_test_list, y_pred_list



from sklearn.metrics import classification_report, confusion_matrix
matrix = confusion_matrix(y_test_list, y_pred_list)
report = classification_report(y_test_list, y_pred_list)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams.update({'font.size': 18})


def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=[rev_index[i] for i in rev_index], columns=[rev_index[i] for i in rev_index])
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap='rocket_r')
    #plt.savefig(filename)
    plt.savefig('confusion_matrix_CXR_Covid-19_{}e_{}_{}_{}.png'.format(EPOCHS, MODEL_NAME,IMG_SIZE,OUTPUT_LAYERS),bbox_inches='tight',pad_inches=0, dpi=200)
    plt.savefig('confusion_matrix_CXR_Covid-19_{}e_{}_{}_{}.eps'.format(EPOCHS, MODEL_NAME,IMG_SIZE,OUTPUT_LAYERS),bbox_inches='tight',pad_inches=0, dpi=200)
    plt.show()

cm_analysis(y_test_list, y_pred_list, [i for i in rev_index] , ymap=None, figsize=(10,10))

