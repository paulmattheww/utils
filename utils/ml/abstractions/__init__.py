


class MLData:
    def __init__(self, df, cat_cols, num_cols, bin_cols, target_col, int_cols=[]):
        self.df = df
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.bin_cols = bin_cols
        self.int_cols = int_cols
        self.target_col = target_col
        self.features = None

    @property
    def X(self):
        if self.features is not None:
            return self.features
        else:
            raise NotImplementedError("Features not specified.  Call obj.make_features().")

    @property
    def y(self):
        return self.df[self.target_col]

    def make_features(self, feature_extraction_func, f_args, f_kwargs):
        """User-provided feature_extraction_func(*args, **kwargs)
        is called and an attribute `features` is set on the instantiation
        of MLData.

        ARGS:
            - feature_extraction_func: function object
            - *f_args: [df]
            - **f_kwargs:
        """
        try:
            setattr(self, "features", feature_extraction_func(*f_args, **f_kwargs))
            print("Features successfully saved to object.")
        except NotImplementedError:
            print(f"{feature_extraction_func} insufficient for feature extraction.")
