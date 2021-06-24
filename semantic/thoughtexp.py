import numpy as np

topic = {}
sentence = "cat dog apple lion NYC love"
tfidf = dict(list(zip(sentence.split(), np.random.rand(6))))




if __name__ == "__main__":
    print(tfidf)