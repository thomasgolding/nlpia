from tfidf.tfidf import TfIdf

docs = ["this is a document", "Sea food is quite nice.", "Kittens are cute."]

categorizer = TfIdf()
categorizer.fit_transform(docs=docs)


def test_tfidf():
    test_docs = ["food", "cute"]
    res = categorizer.get_most_similar(docs=test_docs)
    assert res[0] == docs[1]
    assert res[1] == docs[2]
