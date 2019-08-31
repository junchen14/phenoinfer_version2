import gensim
sentences=gensim.models.word2vec.LineSentence("walks.txt")
model=gensim.models.Word2Vec(sentences,sg=1, min_count=1, size=100, window=3,iter=30,workers=20)
model.save("model_word2vec")
