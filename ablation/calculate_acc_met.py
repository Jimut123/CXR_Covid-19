import numpy as np
import pickle
from sklearn.metrics import confusion_matrix


with open('y_pred_list_100e_InceptionV3_500x500_1024_dropouts-1024-3.pkl','rb') as f:
    y_pred = pickle.load(f)
with open('y_test_list_100e_InceptionV3_500x500_1024_dropouts-1024-3.pkl','rb') as f:
    y_true = pickle.load(f)



#print(confusion_matrix(y_true, y_pred).ravel())




cnf_matrix = confusion_matrix(y_true, y_pred)
#print(cnf_matrix)
#[[1 1 3]
# [3 2 2]
# [1 3 1]]

fp = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
fn = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
tp = np.diag(cnf_matrix)
tn = cnf_matrix.sum() - (fp + fn + tp)

fp = fp.astype(float)
fn = fn.astype(float)
tp = tp.astype(float)
tn = tn.astype(float)

# Sensitivity, hit rate, recall, or true positive rate

specificity = tn / (tn+fp)
acc = (tp+tn) / (tp+tn+fp+fn)
sensitivity = tp / (tp+fn)
precision = tp/(tp+fp)
f1 = 2*precision*sensitivity/(precision*sensitivity)


#TPR = TP/(TP+FN)
# Specificity or true negative rate
#TNR = TN/(TN+FP)
# Precision or positive predictive value
#PPV = TP/(TP+FP)
# Negative predictive value
#NPV = TN/(TN+FN)
# Fall out or false positive rate
#FPR = FP/(FP+TN)
# False negative rate
#FNR = FN/(TP+FN)
# False discovery rate
#FDR = FP/(TP+FP)
# Overall accuracy
#ACC = (TP+TN)/(TP+FP+FN+TN)

acc = sum(acc)/3
sensitivity = sum(sensitivity)/3
specificity = sum(specificity)/3
f1 = sum(f1)/3
precision = sum(precision)/3

print(" Accuracy = ", acc, " Sensitivity = ", sensitivity, " Specificity = ", specificity,"\n",
      " F1 = ", f1, " Precision = ",precision)


