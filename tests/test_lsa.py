from lsa.lsa import LSA

corpus = [
    "cheese bread",
    "wine beer",
    " i enjoy cheese on bread",
    "I drink beer on weekends.",
]

vocabulary = ["cheese", "bread", "wine", "beer"]
topics = [["cheese", "bread"], ["wine", "beer"]]

ntopic = 2
lsa = LSA(ntopic=ntopic, vocabulary=vocabulary)
lsa.fit_transform(docs=corpus)


def test_lsa_number_of_topics():
    assert lsa.docs_topic_vectors.shape[0] == ntopic


def test_lsa_topic_words():
    for topic_words, derived_topic in zip(topics, lsa.topic_words):
        n_words = len(topic_words)
        for w in derived_topic[0:n_words]:
            assert w in topic_words
