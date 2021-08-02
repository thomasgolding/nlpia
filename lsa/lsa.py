from typing import Optional
import numpy as np


from tfidf.tfidf import TfIdf

class LSA:
    def __init__(self, ntopic: int=3, vocabulary: Optional[list]):
        self.ntopic = ntopic
        self.tfidf = TfIdf()
    
    def fit_transform(self, docs: list[str]) -> None:
        self.tfidf.fit_transform(docs = docs)
        docs_m = self.tfidf.get_docs_column_matrix().toarray()
        u, s, vt = np.linalg.svd(docs_m)

        # prepare the document topicvectors
        sigma = np.zeros((self.ntopic, vt.shape[0]))
        for i in range(self.ntopic):
            sigma[i,i] = s[i]

        self.docs_topic_vectors = sigma.dot(vt)

        # prepare the topics
        self.u = u
        self.s = s
        self.vt = vt




        return

    def get_most_similar(self):
        pass

    