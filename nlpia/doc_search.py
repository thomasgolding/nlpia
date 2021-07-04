from tfidf.tfidf import TfIdf


class DocSearch():
    def __init__(self, method: str = "tfidf", similaritymeasure: str = "cosine"):
        self.is_fitted = False
        self.method = method
        self.similaritymeasure = similaritymeasure
        if self.method == "tfidf":
            self.vec = TdIdf()
        else:
            raise ValueError("supported method values: tfidf")

    def fit(self, docs: list[str]):
        if not self.is_fitted:
            self.x = self.vec.fit_transform(docs=docs)
            self.is_fitted = True

    def search(self, query: str):
        if not self.is_fitted:
            raise NotFittedError

        xx = self.vec.transform(docs=[query])
        similiarities = self.calc_similarities(xx, self)

        # find the lowest and list the n loweest docs with most similar




