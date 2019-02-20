import random
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from numpy.random import randint
from sklearn.model_selection import learning_curve

def boxplot_columns_over_groups(df, cols_to_boxplot, unique_groups,
                                grpcol_name, treatment_col,
                                savefig=False, fname='fig.png'):
    plt.clf()
    positions = np.arange(1, df[treatment_col].unique().shape[0]+1)
    ncols_to_plot, ngroups = len(cols_to_boxplot), len(unique_groups)
    fig, axes = plt.subplots(ncols_to_plot, ngroups, figsize=(ngroups*7, ncols_to_plot*6))
    for i, col in enumerate(cols_to_boxplot):
        for j, grp in enumerate(unique_groups):
            in_group = df[grpcol_name] == grp
            _df = df.loc[in_group, [treatment_col, grpcol_name, col]]
            treated = _df[treatment_col]==0
            ax = axes[i, j]
            _df.boxplot(col, by=treatment_col, ax=ax)
            ax.set_title(str(col).replace('_', ' ').title() + ' for ' + str(grp))
            ax.grid(alpha=.3)
            ax.set_xlabel(treatment_col)
            ax.set_ylabel(col)
    sns.despine()
    plt.savefig(fname)
    plt.show()


def random_hex(seed):
    np.random.seed(seed)
    r = lambda: random.randint(0, 255)
    return '#%02X%02X%02X' % (r(),r(),r())


def random_rgb():
    return 'rgb({}, {}, {})'.format(randint(0, 255), randint(0, 255), randint(0, 255))


def plot_learning_curve(mod, X, y, cv, n_jobs, title, ax=None, invert=True):
    '''
    Generates a simple plot of test & training learning curves.
    Inspired from https://github.com/cs109/a-2017/blob/master/Sections/Standard/section_9_student.ipynb
    and from lecture/section.

    Inputs:
    -----------------------------------------------------------------
     mod: model for which learning curve must be plotted
     X: predictor data
     y: true labels
     cv: number cross validation iterations
     n_jobs: number of cores (-1 for all available)
     ax: optional matplotlib Axes object on which to plot

    Outputs:
    -----------------------------------------------------------------
     None: plotted learning curves
    '''
    plt.style.use('seaborn-whitegrid')

    train_sizes, train_scores, test_scores = learning_curve(mod, X=X, y=y_train.values.ravel(), cv=20, n_jobs=-1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if ax == None: fig, ax = plt.subplots(figsize=(12, 7))
    if invert: ax.invert_yaxis()

    ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='training score')
    ax.plot(train_sizes, test_scores_mean, 'o-', color='g', label='test score')
    ax.set_xlabel('Training Examples')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.grid(alpha=0.5)
    sns.despine(bottom=True, left=True)
    ax.legend(loc='best')
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.show()


def plot_accuracy_loss(history_obj, width=16):
    '''
    Leans on Keras' history.history object to visualize fit of model.
    '''
    plt.clf()
    loss = history_obj['loss']
    val_loss = history_obj['val_loss']
    d_loss = np.subtract(loss, val_loss)
    acc = history_obj['acc']
    val_acc = history_obj['val_acc']
    d_acc = np.subtract(acc, val_acc)
    epochs = range(1, len(loss)+1)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(width, 6))
    ax[0].plot(epochs, loss, 'g', label='Training Loss', linestyle='--')
    ax[0].plot(epochs, val_loss, 'b', label='Validation Loss', linestyle='-.')
    ax[0].set_title('Training & Validation Loss')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Loss')
    ax[0].grid(alpha=0.3)
    ax[0].legend(loc='best')
    ax[1].plot(epochs, acc, 'g', label='Training Accuracy', linestyle='--')
    ax[1].plot(epochs, val_acc, 'b', label='Validation Accuracy', linestyle='-.')
    ax[1].set_title('Training & Validation Accuracy')
    ax[1].set_xlabel('')
    ax[1].set_ylabel('Accuracy')
    ax[1].grid(alpha=0.3)
    ax[1].legend(loc='best')
    sns.despine()
    plt.suptitle('Training vs. Validation of Sequential Network Model Over Various Epochs')

    fig2, ax2 = plt.subplots(1, 2, figsize=(width, 3))
    ax2[0].plot(epochs, d_loss, c='black', label='train_loss - val_loss')
    ax2[0].grid(alpha=0.3)
    ax2[0].set_xlabel('Epochs')
    ax2[0].set_ylabel('Loss Differential (Train-Val)')
    ax2[0].legend(loc='best')
    ax2[0].axhline(0, c='black', linestyle=':')
    ax2[0].set_title('Difference of Curves Above')
    ax2[1].plot(epochs, d_acc, c='black', label='train_accuracy - val_accuracy')
    ax2[1].grid(alpha=0.3)
    ax2[1].set_xlabel('Epochs')
    ax2[1].set_ylabel('Accuracy Differential (Train-Val)')
    ax2[1].legend(loc='best')
    ax2[1].axhline(0, c='black', linestyle=':')
    ax2[1].set_title('Difference of Curves Above')
    sns.despine()
    plt.show()
