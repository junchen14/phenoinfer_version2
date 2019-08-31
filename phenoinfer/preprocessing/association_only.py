import gensim
import random
import numpy as np
import pickle as pkl
from scipy.stats import rankdata

type=["union"]
output_directory="../data/boost_traindata_only/"

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


    train_data = open(output_directory+tp+"_train.txt", "w")
    train_data_group = open(output_directory+tp + "_train_group.txt", "w")

    test_data_path = output_directory+tp+"test.txt"
    test_data_store=dict()


    feature_id=1
    test_feature_id=1

    gene_list=[gene for gene in gene_set]

    random.shuffle(gene_list)
    for disease in disease_gene.keys():


        disease_vec=model[disease]

        if disease in train_disease:
            group_number = 0
            genes=disease_gene[disease]

            # generate the positive data
            for gene in genes:

                gene_vec=model[gene]
                gene_disease_vec=np.append(gene_vec,disease_vec,0)

                group_number+=1
                train_data.write("1 ")
                for ind in range(len(gene_disease_vec)):
                    if ind != len(gene_disease_vec) - 1:
                        train_data.write(str(feature_id) + ":" + str(gene_disease_vec[ind]) + " ")
                    else:
                        train_data.write(str(feature_id) + ":" + str(gene_disease_vec[ind]) + "\n")
                        feature_id = 0
                    feature_id += 1

            # generate the negative data

            random_index_list=[]
            while(len(random_index_list)<40):
                random_index=random.choice(range(len(gene_list)))
                if (random_index_list not in random_index_list):
                    if(gene_list[random_index] not in genes):
                        random_index_list.append(random_index)

            for index in random_index_list:
                gene=gene_list[index]
                gene_vec=model[gene]
                gene_disease_vec=np.append(gene_vec,disease_vec,0)


                group_number+=1
                train_data.write("0 ")
                for ind in range(len(gene_disease_vec)):
                    if ind != len(gene_disease_vec) - 1:
                        train_data.write(str(feature_id) + ":" + str(gene_disease_vec[ind]) + " ")
                    else:
                        train_data.write(str(feature_id) + ":" + str(gene_disease_vec[ind]) + "\n")
                        feature_id = 0
                    feature_id += 1
            train_data_group.write(str(group_number) + "\n")

        else:

            genes=disease_gene[disease]
            test_data_store[disease]=genes


    train_data.close()
    train_data_group.close()
    with open(test_data_path,"wb") as f:
        pkl.dump(test_data_store,f)

