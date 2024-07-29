from dataclasses import dataclass

from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class DictionaryTransformer(BaseEstimator, TransformerMixin):
    """Transforms a column of values to a dictionary mapping.

    Notes:
        While we could have used a simple dictionary, this class
        is created to be consistent with the scikit-learn API.
    """

    mapping: dict = None

    def fit(self, X, y=None):
        self.mapping = X
        return self

    def transform(self, X):
        return [self.mapping[x] for x in X]
