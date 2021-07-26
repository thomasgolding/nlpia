from tfidf.tfidf import TfIdf


class NotFittedError(Exception):
    pass


class DocSearch:
    def __init__(self, method: str = "tfidf", similaritymeasure: str = "cosine"):
        self.is_fitted = False
        self.method = method
        self.similaritymeasure = similaritymeasure
        if self.method == "tfidf":
            self.vec = TfIdf()
        else:
            raise ValueError("supported method values: tfidf")

    def fit(self, docs: list[str]):
        if not self.is_fitted:
            self.vec.fit_transform(docs=docs)
            self.is_fitted = True

    def search(self, query: str) -> str:
        if not self.is_fitted:
            raise NotFittedError

        xx = self.vec.get_most_similar(docs=[query])
        return xx[0]
