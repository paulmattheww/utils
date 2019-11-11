import re
import string
from time import time

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from tld import get_tld
from sklearn.decomposition import PCA
from gensim import models
from gensim.models import doc2vec
from gensim.sklearn_api.phrases import PhrasesTransformer
from gensim.models import Phrases
from gensim.corpora import Dictionary
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from six import string_types
from textblob import TextBlob

class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    """Part of speech tagger using NLTK from the blog
    https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html
    for slotting into sklearn pipeline
    """

    def __init__(self, stopwords=None, punct=None,
                 lower=True, strip=True):
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = stopwords or set(sw.words('english'))
        self.punct      = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)



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


class StemmedTfidfVectorizer(TfidfVectorizer):

    def __init__(self, *args, **kwargs):
        super(StemmedTfidfVectorizer, self).__init__(*args, **kwargs)
        self.stemmer = SnowballStemmer('english', ignore_stopwords=False)

    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (self.stemmer.stem(word) for word in analyzer(doc.replace('\n', ' ')))


class ColumnExtractor(TransformerMixin):
    """Returns a subset of df"""
    def __init__(self, features=None):
        self.features = features
        self.col_inds = None

    def transform(self, X, y=None):
        """Subset by column names or by indexes"""
        if isinstance(X, pd.DataFrame):
            return X[self.features].values
        return X[:, self.col_inds]

    def fit(self, X, y=None):
        """find column indeces"""
        self.col_inds =  [ind for ind, col in enumerate(X.columns)
                          if col in self.features]
        return self


class TextCombiner(TransformerMixin):
    """Concat all columns together"""
    def __init__(self):
        pass

    def transform(self, X, y=None):
        """return a series of the combined columns"""
        concat = X[:, 0]
        for ind in range(1, X.shape[1]):
            concat += ' ' + X[:, ind]
        return concat

    def fit(self, X, y=None):
        """No fitting, interface conforming"""
        return self


class EnsureStrings(TransformerMixin):
    """Make all columns strings"""
    def __init__(self):
        self.vect_str_func = np.vectorize(str)

    def transform(self, X, y=None):
        return self.vect_str_func(X)

    def fit(self, X, y=None):
        return self


class MultiColumnLabelEncoder(TransformerMixin):
    """A Label encoder than handles all columns in passed dataframe"""
    def __init__(self):
        self.individual_encoders = []

    def fit(self, X, y=None):
        """no op interface conforming"""
        return self

    def transform(self,X):
        """transform all columns with encoder"""
        return np.apply_along_axis(func1d=LabelEncoder().fit_transform, axis=0, arr=X)


class EnsureNoNans(TransformerMixin):
    """Returns a subset of df"""
    def __init__(self, filler=0):
        self.filler = filler

    def transform(self, X, y=None):
        """Subset by column names or by indexes"""
        return np.nan_to_num(X)

    def fit(self, X, y=None):
        """find column indeces"""
        return self

class ReplaceNans(TransformerMixin):
    """numpy array nans to empty"""
    def __init__(self, filler=""):
        self.filler = filler

    def transform(self, X, y=None):
        """use filler to replace nans"""
        return np.vectorize(self._replacer)(X)

    def fit(self, X, y=None):
        return self

    def _replacer(self, x):
        if x is np.nan:
            return self.filler
        return x

class GetTotalTextLength(TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None):
        return np.array([len(x.split(' ')) for x in X]).reshape(-1, 1)

    def fit(self, X, y=None):
        return self

class MakeTaggedDocument(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [TaggedDocument(words=word_tokenize(doc.lower()), tags=[ind])
                            for ind, doc
                            in enumerate(X)
                           ]

class DateFeaturizer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        return np.array([[x.year, x.month, x.day, x.weekofyear]
                         for x in
                         X])


    def get_day_month_year(df,data_col):
        """ Give pandas datetime in dataframe extracts the
        day of the week the month and the year as feature"""
        # Date data extraction

        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['DayOfMonth'] = df.index.day
        df['WeekOfYear'] = df.index.weekofyear
        return df


class D2VTransformer(TransformerMixin, BaseEstimator):
    """Base Doc2Vec module, wraps :class:`~gensim.models.doc2vec.Doc2Vec`.
    This model based on `Quoc Le, Tomas Mikolov: "Distributed Representations of Sentences and Documents"
    <https://cs.stanford.edu/~quocle/paragraph_vector.pdf>`_.
    """
    def __init__(self, dm_mean=None, dm=1, dbow_words=0, dm_concat=0, dm_tag_count=1, docvecs=None,
                 docvecs_mapfile=None, comment=None, trim_rule=None, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001, hs=0, negative=5, cbow_mean=1,
                 hashfxn=hash, iter=5, sorted_vocab=1, batch_words=10000):
        """
        Parameters
        ----------
        dm_mean : int {1,0}, optional
            If 0, use the sum of the context word vectors. If 1, use the mean. Only applies when `dm_concat=0`.
        dm : int {1,0}, optional
            Defines the training algorithm. If `dm=1` - distributed memory (PV-DM) is used.
            Otherwise, distributed bag of words (PV-DBOW) is employed.
        dbow_words : int {1,0}, optional
            If set to 1 - trains word-vectors (in skip-gram fashion) simultaneous with DBOW
            doc-vector training, If 0, only trains doc-vectors (faster).
        dm_concat : int {1,0}, optional
            If 1, use concatenation of context vectors rather than sum/average.
            Note concatenation results in a much-larger model, as the input is no longer the size of one
            (sampled or arithmetically combined) word vector, but the size of the tag(s) and all words
            in the context strung together.
        dm_tag_count : int, optional
            Expected constant number of document tags per document, when using dm_concat mode.
        docvecs : :class:`~gensim.models.keyedvectors.Doc2VecKeyedVectors`
            A mapping from a string or int tag to its vector representation.
            Either this or `docvecs_mapfile` **MUST** be supplied.
        docvecs_mapfile : str, optional
            Path to a file containing the docvecs mapping. If `docvecs` is None, this file will be used to create it.
        comment : str, optional
            A model descriptive comment, used for logging and debugging purposes.
        trim_rule : function ((str, int, int) -> int), optional
            Vocabulary trimming rule that accepts (word, count, min_count).
            Specifies whether certain words should remain in the vocabulary (:attr:`gensim.utils.RULE_KEEP`),
            be trimmed away (:attr:`gensim.utils.RULE_DISCARD`), or handled using the default
            (:attr:`gensim.utils.RULE_DEFAULT`).
            If None, then :func:`gensim.utils.keep_vocab_item` will be used.
        size : int, optional
            Dimensionality of the feature vectors.
        alpha : float, optional
            The initial learning rate.
        window : int, optional
            The maximum distance between the current and predicted word within a sentence.
        min_count : int, optional
            Ignores all words with total frequency lower than this.
        max_vocab_size : int, optional
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        sample : float, optional
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        seed : int, optional
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`.
            Note that for a **fully deterministically-reproducible run**, you **must also limit the model to
            a single worker thread (`workers=1`)**, to eliminate ordering jitter from OS thread scheduling.
            In Python 3, reproducibility between interpreter launches also requires use of the `PYTHONHASHSEED`
            environment variable to control hash randomization.
        workers : int, optional
            Use this many worker threads to train the model. Will yield a speedup when training with multicore machines.
        min_alpha : float, optional
            Learning rate will linearly drop to `min_alpha` as training progresses.
        hs : int {1,0}, optional
            If 1, hierarchical softmax will be used for model training. If set to 0, and `negative` is non-zero,
            negative sampling will be used.
        negative : int, optional
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
        cbow_mean : int, optional
            Same as `dm_mean`, **unused**.
        hashfxn : function (object -> int), optional
            A hashing function. Used to create an initial random reproducible vector by hashing the random seed.
        iter : int, optional
            Number of epochs to iterate through the corpus.
        sorted_vocab : bool, optional
            Whether the vocabulary should be sorted internally.
        batch_words : int, optional
            Number of words to be handled by each job.
        """
        self.gensim_model = None
        self.dm_mean = dm_mean
        self.dm = dm
        self.dbow_words = dbow_words
        self.dm_concat = dm_concat
        self.dm_tag_count = dm_tag_count
        self.docvecs = docvecs
        self.docvecs_mapfile = docvecs_mapfile
        self.comment = comment
        self.trim_rule = trim_rule

        # attributes associated with gensim.models.Word2Vec
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.iter = iter
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words

    def fit(self, X, y=None):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {iterable of :class:`~gensim.models.doc2vec.TaggedDocument`, iterable of list of str}
            A collection of tagged documents used for training the model.
        Returns
        -------
        :class:`~gensim.sklearn_api.d2vmodel.D2VTransformer`
            The trained model.
        """
        if isinstance(X[0], doc2vec.TaggedDocument):
            d2v_sentences = X
        else:
            d2v_sentences = [doc2vec.TaggedDocument(words, [i]) for i, words in enumerate(X)]
        self.gensim_model = models.Doc2Vec(
            documents=d2v_sentences, dm_mean=self.dm_mean, dm=self.dm,
            dbow_words=self.dbow_words, dm_concat=self.dm_concat, dm_tag_count=self.dm_tag_count,
            docvecs=self.docvecs, docvecs_mapfile=self.docvecs_mapfile, comment=self.comment,
            trim_rule=self.trim_rule, vector_size=self.size, alpha=self.alpha, window=self.window,
            min_count=self.min_count, max_vocab_size=self.max_vocab_size, sample=self.sample,
            seed=self.seed, workers=self.workers, min_alpha=self.min_alpha, hs=self.hs,
            negative=self.negative, cbow_mean=self.cbow_mean, hashfxn=self.hashfxn,
            epochs=self.iter, sorted_vocab=self.sorted_vocab, batch_words=self.batch_words
        )
        return self

    def transform(self, docs):
        """Infer the vector representations for the input documents.
        Parameters
        ----------
        docs : {iterable of list of str, list of str}
            Input document or sequence of documents.
        Returns
        -------
        numpy.ndarray of shape [`len(docs)`, `size`]
            The vector representation of the `docs`.
        """
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # The input as array of array
        if isinstance(docs[0], string_types):
            docs = [docs]
        vectors = [self.gensim_model.infer_vector(doc[0]) for doc in docs] ## CHANGEDFROMSOURCE
        return np.reshape(np.array(vectors), (len(docs), self.gensim_model.vector_size))

class GenericTokenizer(BaseEstimator, TransformerMixin):
    '''Crappy tokenizer that just splits the str on spaces'''
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return [str(x).split() for x in X]

class GensimBigramTransformer(BaseEstimator, TransformerMixin):
    '''Uses Gensim Phrases to select ONLY bigrams from docs'''
    def __init__(self, phrases_kwargs=dict(min_count=1, threshold=5)):
        self.phrases_kwargs = phrases_kwargs
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return [list(filter(lambda x: '_' in x, ls)) for ls in
                PhrasesTransformer(**self.phrases_kwargs).fit_transform(X)]

class RecombineBigrams(BaseEstimator, TransformerMixin):
    '''Recombines tokenized bigram list of lists back into list of str'''
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return [' '.join(x) for x in X]


class SpellingCorrectionTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        blob_docs = [TextBlob(doc) for doc in X]
        return [doc.correct() for doc in blob_docs]


class TextNormalizer(TransformerMixin):
    '''All x in X must be of type str'''

    def __init__(self, lowercase=True, punctuation=punctuation,
                stopwords=stop_words, numerics=True):
        self.stopwords = stopwords
        self.punctuation = punctuation
        self.lowercase = lowercase
        self.numerics = numerics

    def drop_newlines(self, X, y=None):
        docs = [doc.replace('\r', ' ') for doc in X]
        docs = [doc.replace('\\n', ' ') for doc in docs]
        docs = [doc.replace('\n', ' ') for doc in docs]
        return docs

    def to_lowercase(self, X, y=None):
        if self.lowercase:
            return [x.lower() for x in X]
        return X

    def drop_punctuation(self, X, y=None):
        if self.punctuation is None:
            return X
        return [x.translate(str.maketrans(' ', ' ', self.punctuation)) for x in X]

    def drop_stopwords(self, X, y=None):
        if len(self.stopwords) == 0:
            return X
        return [' '.join([word for word in doc.split() if word not in self.stopwords]) for doc in X]

    def drop_numerics(self, X, y=None):
        if not self.numerics:
            return [' '.join(word for word in doc.split() if word.isdigit() == False) for doc in X]
#             return [' '.join(word for word in doc.split() if any(char.isdigit() for char in set(word)) == False) for doc in X]
#             return [' '.join(word for word in doc.split() if bool(re.search(r'\d', word)) == False) for doc in X]
        return X

    def drop_repeats(self, X, y=None):
        return [re.sub(r'((.)\2{2,})', ' ', doc) for doc in X]

    def basic_dlp_str(self, text):

        re_dict = dict(basic_ssn_format = [r"\d{3}-\d{2}-\d{4}", " <SSN> "],
                       basic_ssn_nodashes_format = [r"\d{9}", " <SSN> "],
                       basic_ssn_per_format = [r"\d{3}.\d{2}.\d{4}", " <SSN> "],
                       basic_tel10_format = [r"\d{3}-\d{3}-\d{4}", " <TELEPHONE> "],
                       basic_tel10_par_format = [r"\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}", " <TELEPHONE> "],
                       basic_tel10_dot_format = [r"\d{3}.\d{3}.\d{4}", " <TELEPHONE> "],
                       basic_tel10_nodashes_format = [r"\d{10}", " <TELEPHONE> "],
                       basic_tel7_format = [r"\d{3}-\d{4}", " <TELEPHONE> "],
                       basic_tel7_nodashes_format = [r"\d{7}", " <TELEPHONE> "],
                       )
        for k, criteria in re_dict.items():
            pattern = re.compile(criteria[0])
            text = re.sub(pattern, criteria[1], text)
        return text

    def dlp_vector(self, texts):
        return [self.basic_dlp_str(doc) for doc in texts]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        docs = self.drop_repeats(X)
        docs = self.drop_newlines(docs)
        docs = self.drop_punctuation(docs)
        docs = self.to_lowercase(docs)
        docs = self.drop_stopwords(docs)
        docs = self.dlp_vector(docs)
        docs = self.drop_numerics(docs)
        return docs


class DocumentCharacteristicTransformer(TransformerMixin):
    def __init__(self, lang_model='en_core_web_sm', drop_numerics=True, drop_datetimes=True):
        self.lang_model = lang_model
        self.drop_numerics = drop_numerics
        self.drop_datetimes = drop_datetimes
    def fit(self, X, y=None):
        return self
    def spacy_profile(self, X, y=None):
        '''
        Each record has this information at each document to work off of:
                           dep ent_iob ent_type is_alpha is_stop     lemma    pos shape  \
            \n                       O             False   False        \n  SPACE    \n
            '            punct       O             False   False         '  PUNCT     '
            workwell      amod       O              True   False  workwell    ADJ  xxxx
            patient       amod       O              True   False   patient    ADJ  xxxx
            china     compound       O              True   False     china   NOUN  xxxx

                      tag      text  text_eq_lemma
            \n        _SP        \n           True
            '          ``         '           True
            workwell   JJ  workwell           True
            patient    JJ   patient           True
            china      NN     china           True

        The output array is in this format (using identical docs below):
                    pct_not_in_dictionary  pct_in_dictionary       ADJ       ADP      ADV  \
            0  1.0                    0.0                1.0  0.164087  0.006192  0.03096
            1  1.0                    0.0                1.0  0.164087  0.006192  0.03096
            2  1.0                    0.0                1.0  0.164087  0.006192  0.03096
            3  1.0                    0.0                1.0  0.164087  0.006192  0.03096
            4  1.0                    0.0                1.0  0.164087  0.006192  0.03096

                   INTJ      NOUN      PART      PRON    PROPN     PUNCT     SPACE  \
            0  0.006192  0.517028  0.003096  0.009288  0.01548  0.006192  0.086687
            1  0.006192  0.517028  0.003096  0.009288  0.01548  0.006192  0.086687
            2  0.006192  0.517028  0.003096  0.009288  0.01548  0.006192  0.086687
            3  0.006192  0.517028  0.003096  0.009288  0.01548  0.006192  0.086687
            4  0.006192  0.517028  0.003096  0.009288  0.01548  0.006192  0.086687

                   VERB        X  n_unique_words  n_words  total_to_unique
            0  0.123839  0.03096           190.0    314.0         1.652632
            1  0.123839  0.03096           190.0    314.0         1.652632
            2  0.123839  0.03096           190.0    314.0         1.652632
            3  0.123839  0.03096           190.0    314.0         1.652632
            4  0.123839  0.03096           190.0    314.0         1.652632
        '''
        nlp = spacy.load(self.lang_model)
        docs = dict()
        for i, x in enumerate(X):
            doc = nlp(x)
            pos = dict()
            for token in doc:
                pos[token] = dict(text=token.text, lemma=token.lemma_, pos=token.pos_,
                                  tag=token.tag_, dep=token.dep_, shape=token.shape_,
                                  is_alpha=token.is_alpha, is_stop=token.is_stop,
                                  ent_type=token.ent_type_, ent_iob=token.ent_iob_,
                                  token_in_vocab=str(int((token.text in nlp.vocab) or (token.orth in nlp.vocab))))

            # compile the data tags
            pos_df = pd.DataFrame(pos).T
            pos_df['text_eq_lemma'] = pos_df.text == pos_df.lemma

            # drop numerics & datetimes
            if self.drop_numerics:
                pos_df = pos_df.loc[(pos_df.pos != 'NUM') | (pos_df.ent_type != 'CARDINAL')]
            if self.drop_datetimes:
                pos_df = pos_df.loc[~pos_df.ent_type.isin(['DATE', 'TIME'])]

            # compile
            a = (pos_df.pos.value_counts() / pos_df.shape[0]).to_dict()
            b = (pos_df.ent_type.value_counts() / pos_df.shape[0]).to_dict()
            c = (pos_df.token_in_vocab.value_counts() / pos_df.shape[0]).to_dict()
            if '0' not in c.keys():
                c['0'] = 0.
            d = {'n_unique_words': len(set(x.split())), 'n_words': len(x.split())}
            d['total_to_unique'] = d['n_words'] / d['n_unique_words']
            docs[i] = dict(**a, **b, **c, **d)
        docs_df = pd.DataFrame(docs).T.fillna(0.)
        docs_df.rename(columns={'0': 'pct_not_in_dictionary', '1': 'pct_in_dictionary'}, inplace=True)
        return docs_df

    def transform(self, X, y=None):
        return self.spacy_profile(X)


class SpacyTransformer(TransformerMixin):
    def __init__(self, lang_model='en', custom_stopwords=[],
                 phrases_kwargs=dict(min_count=3, threshold=5)):
        self.lang_model = lang_model
        self.nlp = spacy.load(lang_model)
        self.custom_stopwords = custom_stopwords
        self.phrases_kwargs = phrases_kwargs
        if len(custom_stopwords) > 0:
            for stopword in custom_stopwords:
                lexeme = self.nlp.vocab[stopword]
                lexeme.is_stop = True
    def basic_cleaning(self, X, y=None):
        docs = list()
        for doc in X:
            doc, out_doc = self.nlp(doc), list()
            for word in doc:
                if not word.is_stop and not word.is_punct and not word.like_num:
                    out_doc.append(word.lemma_)
            docs.append(out_doc)
        return docs
    def extract_bigrams(self, X, y=None):
        return PhrasesTransformer(**self.phrases_kwargs).fit_transform(X)
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        texts = self.basic_cleaning(X)
        texts = self.extract_bigrams(texts)
        setattr(self, 'text', texts)
        setattr(self, 'dictionary', Dictionary(texts))
        setattr(self, 'corpus', [self.dictionary.doc2bow(text) for text in texts])
        return [' '.join(text) for text in texts]


class TextNormalizer(TransformerMixin):
    '''All x in X must be of type str'''

    def __init__(self, lowercase=True, punctuation=punctuation,
                stopwords=stop_words, numerics=True):
        self.stopwords = stopwords
        self.punctuation = punctuation
        self.lowercase = lowercase
        self.numerics = numerics

    def drop_newlines(self, X, y=None):
        docs = [doc.replace('\r', ' ') for doc in X]
        docs = [doc.replace('\\n', ' ') for doc in docs]
        docs = [doc.replace('\n', ' ') for doc in docs]
        return docs

    def to_lowercase(self, X, y=None):
        if self.lowercase:
            return [x.lower() for x in X]
        return X

    def drop_punctuation(self, X, y=None):
        if self.punctuation is None:
            return X
        return [x.translate(str.maketrans(' ', ' ', self.punctuation)) for x in X]

    def drop_stopwords(self, X, y=None):
        if len(self.stopwords) == 0:
            return X
        return [' '.join([word for word in doc.split() if word not in self.stopwords]) for doc in X]

    def drop_numerics(self, X, y=None):
        if not self.numerics:
            return [' '.join(word for word in doc.split() if word.isdigit() == False) for doc in X]
        return X

    def drop_repeats(self, X, y=None):
        return [re.sub(r'((.)\2{2,})', ' ', doc) for doc in X]

    def basic_dlp_str(self, text):

        re_dict = dict(basic_ssn_format = [r"\d{3}-\d{2}-\d{4}", " "],
                       basic_ssn_nodashes_format = [r"\d{9}", " "],
                       basic_ssn_per_format = [r"\d{3}.\d{2}.\d{4}", " "],
                       basic_tel10_format = [r"\d{3}-\d{3}-\d{4}", " "],
                       basic_tel10_par_format = [r"\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}", " "],
                       basic_tel10_dot_format = [r"\d{3}.\d{3}.\d{4}", " "],
                       basic_tel10_nodashes_format = [r"\d{10}", " "],
                       basic_tel7_format = [r"\d{3}-\d{4}", " "],
                       basic_tel7_nodashes_format = [r"\d{7}", " "],
                       )
        for k, criteria in re_dict.items():
            pattern = re.compile(criteria[0])
            text = re.sub(pattern, criteria[1], text)
        return text

    def dlp_vector(self, texts):
        return [self.basic_dlp_str(doc) for doc in texts]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        docs = self.drop_repeats(X)
        docs = self.drop_newlines(docs)
        docs = self.drop_punctuation(docs)
        docs = self.to_lowercase(docs)
        docs = self.drop_stopwords(docs)
        docs = self.dlp_vector(docs)
        docs = self.drop_numerics(docs)
        return docs
