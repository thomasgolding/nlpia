import numpy as np
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from nlpia.lemmatizer import Lemmatizer


class TfIdf:
    def __init__(self):
        self.lemmatizer = Lemmatizer()
        self.vectorizer = TfidfVectorizer(smooth_idf=True)

    def get_most_similar(self, docs: list[str]) -> list[str]:
        if len(docs) == 0:
            return []

        m = self.docs_matrix
        r = self._transform(docs=docs)
        mnorm = np.sqrt(m.multiply(m).sum(axis=1))
        rnorm = np.sqrt(r.multiply(r).sum(axis=1).transpose())

        prod = m.dot(r.transpose())
        normprod = prod / mnorm / rnorm

        rr = normprod.argmax(axis=0)
        rrflat = rr.getA1()
        bestfit_docs = [self.docs[el] for el in rrflat]
        return bestfit_docs

    def fit_transform(self, docs: list[str]) -> None:
        self.docs = docs
        self.docs_lemmas = self.lemmatizer.lemmatize(docs=docs)
        self.docs_matrix = self.vectorizer.fit_transform(raw_documents=self.docs_lemmas)
        return

    def _transform(self, docs: list[str]) -> csr_matrix:
        docs_lemmas = self.lemmatizer.lemmatize(docs=docs)
        x = self.vectorizer.transform(raw_documents=docs_lemmas)
        return x
