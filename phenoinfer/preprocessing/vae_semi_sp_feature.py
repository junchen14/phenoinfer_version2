import gensim
import random
import numpy as np
import pickle as pkl
from scipy.stats import rankdata
'''
this is to generate some data features for vae and generate the intermediate result with lable 0.5
'''


type=["union"]
output_directory="../data/vae/"
def generate_train_data():
    for tp in type:
        print(tp)
        with open("../data/"+tp+"_disease_gene.pkl","rb") as f:
            disease_gene=pkl.load(f)
        with open("../data/"+tp+"_gene_set.pkl","rb") as f:
            gene_set=pkl.load(f)

        model=gensim.models.Word2Vec.load("../opamodel/"+tp+"_association_only.model")

        disease_set=[disease for disease in disease_gene.keys()]
        random.shuffle(disease_set)
        train_disease=disease_set[:int(len(disease_set)*0.85)]

        total_disease=open(output_directory+tp+"_total_disease.txt","w")
        for disease in disease_gene.keys():
            disease_embed=model[disease]
            for gene in disease_gene[disease]:
                gene_embed=model[gene]
                disease_gene_vec=np.append(disease_embed,gene_embed,0)

                for index in range(len(disease_gene_vec)):
                    if index !=len(disease_gene_vec)-1:
                        total_disease.write(str(disease_gene_vec[index])+" ")
                    else:
                        total_disease.write(str(disease_gene_vec[index])+"\n")
        total_disease.close()


        train_data_path = output_directory+tp+"train.pkl"
        positive_disease_gene = []
        test_data_path = output_directory+tp+"test.pkl"
        test_data_store=dict()

        train_disease_path=output_directory+tp+"train_disease.pkl"
        gene_list=[gene for gene in gene_set]


        for disease in disease_gene.keys():

            if disease in train_disease:
                genes=disease_gene[disease]
                for gene in genes:
                    gene_vec=model[gene]
                    disease_vec=model[disease]
                    gene_disease_vec=np.append(disease_vec,gene_vec,0)
                    positive_disease_gene.append(gene_disease_vec)

            else:
                genes=disease_gene[disease]
                test_data_store[disease]=genes

        random.shuffle(positive_disease_gene)
        print("the length of positive disease gene: ",len(positive_disease_gene))

        with open(train_data_path,'wb') as f:
            pkl.dump(positive_disease_gene,f)

        with open(test_data_path,"wb") as f:
            pkl.dump(test_data_store,f)

        with open(train_disease_path,"wb") as f:
            pkl.dump(train_disease,f)


generate_train_data()