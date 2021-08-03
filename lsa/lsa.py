from typing import Optional

import numpy as np

from tfidf.tfidf import TfIdf


class LSA:
    def __init__(self, ntopic: int = 3, vocabulary: Optional[list[str]] = None):
        self.ntopic = ntopic
        self.vocabulary = vocabulary
        self.tfidf = TfIdf(vocabulary=vocabulary)

    def fit_transform(self, docs: list[str]) -> None:
        self.tfidf.fit_transform(docs=docs)
        docs_m = self.tfidf.get_docs_column_matrix().toarray()
        u, s, vt = np.linalg.svd(docs_m)

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

        return

    def print_topics(self, n_words: int = 3):
        for i_topic in range(self.ntopic):
            print(f"Topic {i_topic}")
            for i_word in range(n_words):
                m = f"{self.topic_weight[i_topic][i_word].round(3):8} --- {self.topic_words[i_topic][i_word]}"
                print(m)
            print()

        return

    def get_most_similar(self):
        pass
