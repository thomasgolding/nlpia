import spacy

class Lemmatizer:
    def __init__(self, docs: list[str]):
        self.nlp = spacy.load("en_core_web_sm")
        self.docs = docs
        self.spacy_docs = [self.nlp(el) for el in docs]
        self.docs_lemma = [
            " ".join(
                [
                    el.lemma_ for el in xdoc
                ]
            )
            for xdoc in self.spacy_docs
        ]



