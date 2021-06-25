import spacy


class Lemmatizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def lemmatize(self, docs: list[str]):
        spacy_docs = [self.nlp(el) for el in docs]
        docs_lemma = [" ".join([el.lemma_ for el in xdoc]) for xdoc in spacy_docs]
        return docs_lemma
