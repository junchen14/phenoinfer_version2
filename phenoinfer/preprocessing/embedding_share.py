import gensim
import pickle as pkl


gene_embedding=dict()
disease_embedding=dict()
tp="union"

opamodel=gensim.models.Word2Vec.load("../opamodel/union.model")
with open("../data/" + tp + "_disease_gene.pkl", "rb") as f:
    disease_gene = pkl.load(f)
with open("../data/" + tp + "_gene_set.pkl", "rb") as f:
    gene_set = pkl.load(f)

disease_set=set()
for disease in disease_gene.keys():
    disease_set.add(disease)


for disease in disease_set:
    disease_embedding[disease]=opamodel[disease]
for gene in gene_set:
    gene_embedding[gene]=opamodel[gene]


with open("../embedding_share/disease_embedding.pkl","wb") as f:
    pkl.dump(disease_embedding,f)
with open("../embedding_share/gene_embedding.pkl","wb") as f:
    pkl.dump(gene_embedding,f)

with open('../embedding_share/disease_gene_association.pkl',"wb") as f:
    pkl.dump(disease_gene,f)
print(len(disease_embedding))
print(len(gene_embedding))