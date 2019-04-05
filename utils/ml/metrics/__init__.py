from sklearn.metrics import confusion_matrix
from pandas import DataFrame
import pandas as pd
import numpy as np

def analyze_binary_prediction_probabilities(model, X, y, bins=10):
    '''Compiles predictions vs. actuals for a binary model
    that has a model.predict_proba() method.  Bins the prediction
    probabilities for further analysis and tags correct answers.

    ARGS:     model, X, y
    KWARGS:   bins
    RETURNS:  DataFrame
    '''
    pred_df = pd.DataFrame({'prediction': model.predict(X), 'label': y})
    pred_df['prediction_probability'] = model.predict_proba(X)[:, 1]
    pred_df['probability_bin'] = pd.cut(pred_df['prediction_probability'], bins=bins)
    pred_df['correct'] = pred_df.prediction == pred_df.label

    return pred_df

def binary_confusion_matrix(y, y_hat, as_pct=False, verbose=True):
    cm = pd.DataFrame(confusion_matrix(y, y_hat),
                      columns=['(+) actual', '(-) actual'],
                      index=['(+) predicted', '(-) predicted'])
    if as_pct:
        cm = cm / cm.sum().sum()

    P = cm['(+) actual'].sum()
    N = cm['(-) actual'].sum()
    total = P + N
    TP = cm.loc['(+) predicted', '(+) actual']
    FP = cm.loc['(+) predicted', '(-) actual']
    TN = cm.loc['(-) predicted', '(-) actual']
    FN = cm.loc['(-) predicted', '(+) actual']
    TPR = TP / (TP + FN)          # recall/sensitivity
    TNR = TN / (TN + FP)   # specificity
    FPR = FP / (FP + TN)   # fall-out
    FNR = FN / (FN + TP)   # miss rate
    PPV = TP / (TP + FP)   # precision
    NPV = TN / (TN + FN)   # neg predictive value

    if verbose:
        print('''
        Condition Positive:                        %i
        Condition Negative:                        %i
        Total Observations:                        %i

        True Positive:                             %i
        True Negative:                             %i
        False Positive:                            %i
        False Negative                             %i

        True Positive Rate (recall):               %.2f%%
        True Negative Rate (specificity):          %.2f%%
        False Positive Rate (fall-out):            %.2f%%
        False Negative Rate (miss rate):           %.2f%%

        Positive Predictive Value (precision):     %.2f%%
        Negative Predictive Value:                 %.2f%%
        ''' %(P, N, total,
             TP, TN, FP, FN,
             TPR*100, TNR*100, FPR*100, FNR*100,
             PPV*100, NPV*100))

    metrics = {'P': P, 'N': N, 'total': total,
              'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
              'TPR': TPR, 'TNR': TNR, 'FPR': FPR, 'FNR': FNR, 'PPV': PPV, 'NPV': NPV}

    return cm, metrics



def multiclass_confusion_matrix(y, yhat, model_name='unspecified',
                               verbose=1):
    '''
    Inputs:
    ------------------------------------------------------
    y: true labels
    yhat: predicted labels
    model_name: name of model for printing

    Outputs:
    ------------------------------------------------------
    cm: confusion matrix (easily readable)
    metrics: dict of metrics on multiclass classification
    '''
    # organize confusion matrix from sklearn into readable format
    sk_confusion_matrix = confusion_matrix(y, yhat).transpose()#; print(sk_confusion_matrix)

    # put in pd.DataFrame and add names
    cm = DataFrame(sk_confusion_matrix)
    IX = ['Test_' + str(i+1) for i in cm.index]
    COLS = ['Condition_' + str(i+1) for i in cm.columns]
    cm.columns, cm.index = COLS, IX

    # add totals
    cm['Total'] = cm.sum(axis=1)
    cm.loc['Total'] = cm.sum(axis=0)

    # get performance scores
    N = cm.loc['Total', 'Total']
    TP = np.diag(cm.loc[IX, COLS]).sum()
    ACC = np.divide(TP, N)
    MCR = 1 - ACC

    metrics = {'accuracy':ACC, 'misclassification':MCR}

    if verbose:
        print('''
        Confusion Matrix for Model: %s
        ------------------------------------------------------''' %model_name)
        print(cm)
        print('''
        Metrics for Model: %s
        ------------------------------------------------------
        Accuracy Rate = %.5f
        Misclassification Rate = %.5f
        ''' %(model_name, ACC, MCR))
        return None

    return cm, metrics
