from sklearn.feature_extraction.text import TfidfVectorizer

from nlpia.lemmatizer import Lemmatizer


class TfIdf():
    def __init__(self):
        self.lemmatizer = Lemmatizer()
        self.vectorizer = TfidfVectorizer(smooth_idf=True)
        
    def fit_transform(self, docs: list[str]):
        docs_lemmas = self.lemmatizer.lemmatize(docs=docs)
        x = self.vectorizer.fit_transform(raw_documents=docs_lemmas)
        return x

    def transform(self, docs: list[str]):
        docs_lemmas = self.lemmatizer.lemmatize(docs=docs)
        x = self.vectorizer.transform(raw_documents=docs_lemmas)
        return x


    


        