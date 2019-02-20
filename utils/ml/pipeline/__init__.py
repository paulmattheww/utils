from sklearn.base import BaseEstimator, TransformerMixin

class ColumnExtractor(BaseEstimator, TransformerMixin):
    """Extracts a single column from DataFrame as a Series
    to preserve the methods.
    """
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        """interface conforming for fit_transform"""
        return self

    def transform(self, X):
        """Expects a pd.DataFrame data type"""
        print('EXTRACTING SINGLE COLUMN ...')
        X_new = X[self.col]#.astype(str)
        return X_new

class ColumnExtractor(BaseEstimator, TransformerMixin):
    '''
    Transformer for extracting columns in sklearn pipeline
    '''
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xcols = X[self.cols]
        return Xcols

class LabelEncodeObjects(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, data):
        for c in data.columns:
            if data[c].dtype == 'object':
                lbl = LabelEncoder()
                lbl.fit(list(data[c].values))
                data[c] = lbl.transform(list(data[c].values))
        return data

class NaFiller(BaseEstimator, TransformerMixin):
    def __init__(self, fill_val=-1):
        self.fill_val = fill_val

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        if self.fill_val == 'mean':
            for col in data.columns:
                data.loc[data[col].isna(), col] = data.loc[~data[col].isna(), col].mean()
        else:
            data = data.fillna(self.fill_val)
        return data
