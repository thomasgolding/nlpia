from lsa.lsa import LSA

corpus = [
    "NYC is the big apple",
    "NYC is known as the big apple",
    "I love NYC",
    "I wore a hat to the big apple prty in NYC",
    "Come to NYC, see the big apple",
    "Manhattan is called the big apple",
    "New york is a big city for a small cat.",
    "The lion, a big cat, is the king of the jungle",
    "I love my pet cat.",
    "I love new york city (NYC)",
    "Your dog chased my cat"
]

vocabulary = ["cat", "dog", "apple", "lion", "nyc", "love"]


ntopic=2
lsa = LSA(ntopic=ntopic)
lsa.fit_transform(docs=corpus)


def test_lsa():
    ntopic=2
    lsa = LSA(ntopic=ntopic)
    lsa.fit_transform(docs=corpus)
    assert lsa.docs_topic_vectors.shape[0] == ntopic 


