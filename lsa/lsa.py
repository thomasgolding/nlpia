from typing import Optional

import numpy as np

from tfidf.tfidf import TfIdf


class LSA:
    def __init__(self, ntopic: int = 3, vocabulary: Optional[list[str]] = None):
        self.ntopic = ntopic
        self.vocabulary = vocabulary
        self.tfidf = TfIdf(vocabulary=vocabulary)
        self.fitted = False

    def fit_transform(self, docs: list[str]) -> None:
        self.tfidf.fit_transform(docs=docs)
        self.docs = docs
        docs_m = self.tfidf.get_docs_column_matrix().toarray()
        u, s, vt = np.linalg.svd(docs_m)
        nt = self.ntopic
        self.topics = u[:, 0:nt]

        # prepare the document topicvectors
        sigma = np.zeros((self.ntopic, vt.shape[0]))
        for i in range(self.ntopic):
            sigma[i, i] = s[i]

        self.docs_topic_vectors = sigma.dot(vt)

        # prepare the topics
        vocab = self.tfidf.vectorizer.get_feature_names()
        self.topic_vocabulary_index = [
            np.argsort(-np.abs(u[:, i_topic])) for i_topic in range(self.ntopic)
        ]
        self.topic_words = [
            [vocab[i] for i in ind] for ind in self.topic_vocabulary_index
        ]

        self.topic_weight = [
            [u[i, i_topic] for i in ind]
            for i_topic, ind in enumerate(self.topic_vocabulary_index)
        ]

        self.fitted = True

        return

    def print_topics(self, n_words: int = 3) -> None:
        for i_topic in range(self.ntopic):
            print(f"Topic {i_topic}")
            for i_word in range(n_words):
                m = f"{self.topic_weight[i_topic][i_word].round(3):8} --- {self.topic_words[i_topic][i_word]}"
                print(m)
            print()

        return

    def _transform(self, docs: list[str]) -> np.ndarray:

        docs_tfidf = self.tfidf._transform(docs=docs).toarray().transpose()
        docs_topic, _, _, _ = np.linalg.lstsq(a=self.topics, b=docs_tfidf, rcond=None)

        # error_docs = self.topics.dot(docs)_topic-docs_tfidf
        # reconstructed = self.topics.dot(docs_topic)
        # diff = reconstructed - docs_tfidf
        # error = np.sqrt((diff**2).sum(axis=0))
        return docs_topic

    def get_most_similar(self, docs: list[str]) -> list[str]:
        if not self.fitted:
            return []

        docs_topic = self._transform(docs=docs)
        m = self.docs_topic_vectors
        r = docs_topic

        mnorm = np.sqrt((m * m).sum(axis=0))
        rnorm = np.sqrt((r * r).sum(axis=0))

        prod = m.transpose().dot(r)
        normprod = prod / np.atleast_2d(mnorm).transpose() / rnorm

        rr = normprod.argmax(axis=0)
        bestfit_docs = [self.docs[el] for el in rr]
        return bestfit_docs
