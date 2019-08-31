import gensim
import random
import numpy as np
import pickle as pkl
from scipy.stats import rankdata

type=["union"]
output_directory="../data/vae/"
def generate_train_data():
    for tp in type:
        print(tp)
        with open("../data/"+tp+"_disease_gene.pkl","rb") as f:
            disease_gene=pkl.load(f)
        with open("../data/"+tp+"_gene_set.pkl","rb") as f:
            gene_set=pkl.load(f)

        model=gensim.models.Word2Vec.load("../opamodel/"+tp+".model")

        disease_set=[disease for disease in disease_gene.keys()]
        random.shuffle(disease_set)
        train_disease=disease_set[:int(len(disease_set)*0.85)]


        total_disease=open(output_directory+tp+"_total_disease.txt","w")
        for disease in disease_gene.keys():
            disease_embed=model[disease]
            for index in range(len(disease_embed)):
                if index !=len(disease_embed)-1:
                    total_disease.write(str(disease_embed[index])+" ")
                else:
                    total_disease.write(str(disease_embed[index])+"\n")
        total_disease.close()


        train_data_path = output_directory+tp+"train.pkl"
        positive_disease_gene = []
        test_data_path = output_directory+tp+"test.pkl"
        test_data_store=dict()

        gene_list=[gene for gene in gene_set]


        for disease in disease_gene.keys():

            if disease in train_disease:
                genes=disease_gene[disease]
                for gene in genes:
                    positive_disease_gene.append([model[disease],model[gene]])

            else:
                genes=disease_gene[disease]
                test_data_store[disease]=genes

        random.shuffle(positive_disease_gene)
        print("the length of positive disease gene: ",len(positive_disease_gene))

        with open(train_data_path,'wb') as f:
            pkl.dump(positive_disease_gene,f)

        with open(test_data_path,"wb") as f:
            pkl.dump(test_data_store,f)


generate_train_data()