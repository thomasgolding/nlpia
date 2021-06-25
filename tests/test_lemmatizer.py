from nlpia.lemmatizer import Lemmatizer

wu = "dsafsadfdsa"
docs = [f"These are two sentences, are you sure {wu}?"]
lemmas = ["these", "be", "sentence", wu]

lemmatizer = Lemmatizer()

def test_lemmatized_docs():
    docs_lemmas = lemmatizer.lemmatize(docs=docs)
    for el in lemmas:
        assert el in docs_lemmas[0] 