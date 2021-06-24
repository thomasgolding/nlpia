from nlpia.lemmatizer import Lemmatizer

wu = "dsafsadfdsa"
docs = [f"These are two sentences, are you sure {wu}?"]
lemmas = ["these", "be", "sentence", wu]

lem = Lemmatizer(docs=docs)

def test_lemmatized_docs():
    for el in lemmas:
        assert el in lem.docs_lemma[0] 