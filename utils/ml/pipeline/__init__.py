from sklearn.base import BaseEstimator, TransformerMixin
import re
import string
from time import time
import spacy
from tld import get_tld
from sklearn.decomposition import PCA

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



# CLASS DEFINITIONS
class TextCleaner(TransformerMixin):
    """Text cleaning to slot into sklearn interface"""

    def __init__(self, remove_stopwords=True, remove_urls=True,
                 remove_puncts=True, lemmatize=True, extra_punct='',
                 custom_stopwords=[], custom_non_stopwords = [],
                 verbose=True, parser='big'):
        """
        DESCR:
        INPUT: remove_stopwords - bool - remove is, there, he etc...
               remove_urls - bool - 't www.monkey.com t' --> 't com t'
               remove_punct - bool - all punct and digits gone
               lemmatize - bool - whether to apply lemmtization
               extra_punct - str - other characters to remove
               custom_stopwords - list - add to standard stops
               custom_non_stopwords - list - make sure are kept
               verbose - bool - whether to print progress statements
               parser - str - 'big' or small, one keeps more, and is slower
        OUTPUT: self - **due to other method, not this one
        """
        # Initialize passed Attributes to specify operations
        self.remove_stopwords = remove_stopwords
        self.remove_urls = remove_urls
        self.remove_puncts = remove_puncts
        self.lemmatize = lemmatize

        # Change how operations work
        self.custom_stopwords = custom_stopwords
        self.custom_non_stopwords = custom_non_stopwords
        self.verbose = verbose

        # Set up punctation tranlation table
        self.removals = string.punctuation + string.digits + extra_punct
        self.trans_table = str.maketrans({key: None for key in self.removals})

        #Load nlp model for parsing usage later
        self.parser = spacy.load('en_core_web_sm',
                                 disable=['parser','ner','textcat'])
        #from spacy.lang.en import English
        if parser == 'small':
            self.parser = spacy.load('en')#English()

        #Add custom stop words to nlp
        for word in self.custom_stopwords:
            self.parser.vocab[word].is_stop = True

        #Set custom nlp words to be kept
        for word in self.custom_non_stopwords:
            self.parser.vocab[word].is_stop = False


    def transform(self, X, y=None):
        """take array of docs to clean array of docs"""
        # Potential replace urls with tld ie www.monkey.com to com
        if self.remove_urls:
            start_time = time()
            if self.verbose:
                print("CHANGING URLS to TLDS...  ", end='')
            X = [self.remove_url(doc) for doc in X]
            if self.verbose:
                print(f"{time() - start_time:.0f} seconds")

        # Potentially remove punctuation
        if self.remove_puncts:
            start_time = time()
            if self.verbose:
                print("REMOVING PUNCTUATION AND DIGITS... ", end='')
            X = [doc.lower().translate(self.trans_table) for doc in X]
            if self.verbose:
                print(f"{time() - start_time:.0f} seconds")

        # Using Spacy to parse text
        start_time = time()
        if self.verbose:
            print("PARSING TEXT WITH SPACY... ", end='')
        #X = list(self.nlp.pipe(X))
        X = list(self.parser.pipe(X))
        if self.verbose:
            print(f"{time() - start_time:.0f} seconds")

        # Potential stopword removal
        if self.remove_stopwords:
            start_time = time()
            if self.verbose:
                print("REMOVING STOP WORDS FROM DOCUMENTS... ", end='')
            X = [[word for word in doc if not word.is_stop] for doc in X]
            if self.verbose:
                print(f"{time() - start_time:.0f} seconds")


        # Potential Lemmatization
        if self.lemmatize:
            start_time = time()
            if self.verbose:
                print("LEMMATIZING WORDS... ", end='')
            X = [[word.lemma_ for word in doc] for doc in X]
            if self.verbose:
                print(f"{time() - start_time:.0f} seconds")

        # Put back to normal if no lemmatizing happened
        if not self.lemmatize:
            X = [[str(word).lower() for word in doc] for doc in X]

        # Join Back up
        return [' '.join(lst) for lst in X]


    def fit(self, X, y=None):
        """interface conforming, and allows use of fit_transform"""
        return self


    @staticmethod
    def remove_url(text):
        """
        DESCR: given a url string find urls and replace with top level domain
               a bit lazy in that if there are multiple all are replaced by first
        INPUT: text - str - 'this is www.monky.com in text'
        OUTPIT: str - 'this is <com> in text'
        """
        # Define string to match urls
        url_re = '((?:www|https?)(://)?[^\s]+)'

        # Find potential things to replace
        matches = re.findall(url_re, text)
        if matches == []:
            return text

        # Get tld of first match
        match = matches[0][0]
        try:
            tld = get_tld(match, fail_silently=True, fix_protocol=True)
        except ValueError:
            tld = None

        # failures return none so change to empty
        if tld is None:
            tld = ""

        # make this obvsiouyly an odd tag
        tld = f"<{tld}>"

        # Make replacements and return
        return re.sub(url_re, tld, text)


class TextEncoder(BaseEstimator, TransformerMixin):
    """Uses a defined encoder model to transform text data
    into a latent space to represent its features in a more
    dense format.
    """
    def __init__(self, encoder_model):
        self.encoder_model = encoder_model

    def preprocess_encodings(self, X, num_words=5000, maxlen=5000):
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(X)
        txt = tokenizer.texts_to_sequences(X)
        vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
        txt = pad_sequences(txt, padding='post', maxlen=maxlen)
        return txt

    def fit(self, X, y=None):
        """interface conforming for fit_transform"""
        return self

    def transform(self, X, use_pca=False):
        print('ENCODING FEATURES ...')
        X_new = self.preprocess_encodings(X)
        if use_pca:
            return PCA(n_components=10).fit_transform(self.encoder_model.predict(X_new))
        else:
            return self.encoder_model.predict(X_new)
