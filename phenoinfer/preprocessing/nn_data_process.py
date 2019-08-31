import gensim
import random
import numpy as np
import pickle as pkl

type=["union","intersection","uberon","mp","go"]

for tp in type:
    print(tp)
    with open("../data/"+tp+"_disease_gene.pkl","rb") as f:
        disease_gene=pkl.load(f)
    with open("../data/"+tp+"_gene_set.pkl","rb") as f:
        gene_set=pkl.load(f)
    print(len(gene_set))

    model=gensim.models.Word2Vec.load("../opamodel/"+tp+".model")

    disease_set=[disease for disease in disease_gene.keys()]
    random.shuffle(disease_set)
    train_disease=disease_set[:int(len(disease_set)*0.8)]


    positive_data=[]
    negative_data=[]
    gene_list=[gene for gene in gene_set]

    eval_dict=dict()

    for disease in disease_gene.keys():
        disease_vec=model[disease]
        if disease in train_disease:
            genes=disease_gene[disease]

            # generate the positive data
            for gene in genes:
                gene_vec=model[gene]
                gene_disease_vec=np.append(gene_vec,disease_vec,0)
                positive_data.append(gene_disease_vec)

            # generate the negative data

            random_index_list=[]
            while(len(random_index_list)<50):
                random_index=random.choice(range(len(gene_list)))
                if (random_index_list not in random_index_list):
                    if(gene_list[random_index] not in genes):
                        random_index_list.append(random_index)

            for index in random_index_list:
                gene=gene_list[index]
                gene_vec=model[gene]
                gene_disease_vec=np.append(gene_vec,disease_vec,0)
                negative_data.append(gene_disease_vec)


        else:
            eval_dict[disease]=disease_gene[disease]


    with open("../data/"+tp+"_eval_data.pkl","wb") as f:
        pkl.dump(eval_dict,f)
    with open("../data/"+tp+"_positive_data.pkl","wb") as f:
        pkl.dump(positive_data,f)
    with open("../data/"+tp+"_negative_data.pkl","wb") as f:
        pkl.dump(negative_data,f)
