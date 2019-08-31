import gensim
import gensim.models
import os
import sys
myclasses = str(sys.argv[1])
mywindow= int(sys.argv[2])
mysize= int(sys.argv[3])
mincount=int(sys.argv[4])
model =str (sys.argv[5])
pretrain=str (sys.argv[6])
outfile=str(sys.argv[7])


sentences =gensim.models.word2vec.LineSentence('ontology_corpus.lst') # a memory-friendly iterator

print ("vocabulary is ready \n")
if (model =="sg"):
   ssmodel =gensim.models.Word2Vec(sentences,sg=1,min_count=mincount, size=mysize ,window=mywindow, sample=1e-3,iter=30)
else:
   ssmodel =gensim.models.Word2Vec(sentences,sg=0,min_count=mincount, size=mysize ,window=mywindow)
ssmodel.save(outfile)

#
# mymodel=gensim.models.Word2Vec.load (pretrain)
# sentences =gensim.models.word2vec.LineSentence('ontology_corpus.lst')
# mymodel.build_vocab(sentences, update=True)
# #mymodel =gensim.models.Word2Vec(sentences,sg=0,min_count=0, size=200 ,window=5, sample=1e-3)
# mymodel.train (sentences,total_examples=mymodel.corpus_count, epochs=mymodel.iter)
# #print (len(mymodel.wv.vocab));1
# # Store vectors for each given class
# word_vectors=mymodel.wv

# mymodel.save(outfile)
