import random
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from numpy.random import randint
from sklearn.model_selection import learning_curve
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

def rf_feature_importances(forest, title='', outpath=None, use_top=None, figsize=(13, 8)):
    '''Derive feature importances from an sklearn.ensemble.RandomForestClassifier
    model and plots them in descending order for review and analysis.
    '''
    importances = forest.feature_importances_
    feats = {}
    for feature, importance in zip(X_train.columns, importances):
        feats[feature] = importance

    fig, ax = plt.subplots(figsize=figsize)
    feats = pd.Series(feats).sort_values()
    if use_top is not None:
        feats = feats[-use_top:]
    feats.plot(kind="barh", ax=ax)
    ax = plt.gca()
    sns.despine()
    ax.set_title(title)
    if outpath:
        plt.savefig(outpath)
    plt.show()


def correlation_heatmap(df, cutoff=None, title='', outpath=None):
    df_corr = df.corr('pearson')
    np.fill_diagonal(df_corr.values, 0)
    if cutoff != None:
        for col in df_corr.columns:
            df_corr.loc[df_corr[col].abs() <= cutoff, col] = 0
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(df_corr, ax=ax, cmap='RdBu_r')
    plt.suptitle(title, size=18)
    if outpath == None:
        pass
    else:
        plt.savefig(outpath)
    plt.show()
    return df_corr


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




def plot_tseries_over_group_with_histograms(df, xcol, ycol,
                                            grpcol, title_prepend='{}',
                                            labs=None, x_angle=0, labelpad=60,
                                            window=15, ignore_cols=[]):
    '''
    Function for plotting time series df[ycol] over datetime range df[xcol]
    using the unique_grp_vals contained in df[grpcol].unique().

     - df: pd.DataFrame containing datetime and series to plot
     - xcol: str of column name in df for datetime series
     - ycol: str of column name in df for tseries
     - grpcol: str of column name in df of group over which to plot
     - labs: dict of xlab, ylab
     - title_prepend: str containing "{}" that prepends group names in title

     Example:
         title_prepend = 'Time Series for {}'
         xcol = 'date'
         ycol = 'rolling_15_mean'
         grpcol = 'variable'
         labs = dict(xlab='', ylab='Value')

         plot_tseries_over_group_with_histograms(smooth_df,
                                                 xcol, ycol, grpcol,
                                                 title_prepend, labs,
                                                 x_angle=90,
                                                 ignore_cols=onehot_cols)

    '''
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')

    unique_grp_vals = df[grpcol].unique()
    nrows = len(unique_grp_vals) - len(ignore_cols)
    figsize = (13, 6 * nrows)
    fig, axes = plt.subplots(nrows, 1, figsize=figsize)
    title_prepend_hist = 'Histogram of ' + str(title_prepend)
    j = 0
    for i, grp in enumerate(unique_grp_vals):
        _df = df.loc[df[grpcol] == grp]
        if grp not in ignore_cols:
            _df = df.loc[df[grpcol] == grp]
            ax = axes[j]
            ax.plot(_df[xcol], _df[ycol], alpha=.2, color='black')
            ax.plot(_df[xcol], _df[ycol].rolling(window=window, min_periods=min(5, window)).mean(),
                    alpha=.5, color='r', label='{} period rolling avg'.format(window),
                    linestyle='--')
            longer_window = int(window * 3)
            ax.plot(_df[xcol], _df[ycol].rolling(window=longer_window, min_periods=5).mean(),
                    alpha=.8, color='darkred', label='{} period rolling avg'.format(longer_window),
                    linewidth=2)
            mu, sigma = _df[ycol].mean(), _df[ycol].std()
            ax.axhline(mu, linestyle='--', color='r', alpha=.3)
            ax.axhline(mu - sigma, linestyle='-.', color='y', alpha=.3)
            ax.axhline(mu + sigma, linestyle='-.', color='y', alpha=.3)
            ax.set_title(title_prepend.format(grp))
            ax.legend(loc='best')
            bottom, top = mu - 3*sigma, mu + 3*sigma
            ax.set_ylim((bottom, top))
            if labs is not None:
                ax.set_xlabel(labs['xlab'])
                ax.set_ylabel(labs['ylab'])
            ax.xaxis.labelpad = labelpad
            ax.xaxis.set_minor_locator(months)
            ax.grid(alpha=.1)
            if x_angle != 0:
                for tick in ax.get_xticklabels():
                    tick.set_rotation(x_angle)

            divider = make_axes_locatable(ax)
            axHisty = divider.append_axes('right', 1.2, pad=0.1, sharey=ax)
            axHisty.grid(alpha=.1)
            axHisty.hist(_df[ycol].dropna(), orientation='horizontal', alpha=.5, color='lightgreen', bins=25)
            axHisty.axhline(mu, linestyle='--', color='r', label='mu', alpha=.3)
            axHisty.axhline(mu - sigma, linestyle='-.', color='y', label='+/- two sigma', alpha=.3)
            axHisty.axhline(mu + sigma, linestyle='-.', color='y', alpha=.3)
            axHisty.legend(loc='best')

            j += 1
        else:
            pass

    sns.set_style("whitegrid")
    sns.despine()
    plt.show()
