import gensim
import random
import numpy as np
import pickle as pkl
from scipy.stats import rankdata

type=["union"]
output_directory="../data/boost_bidirectional_sampling_part_gene/"
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


        train_data = open(output_directory+tp+"_train.txt", "w")
        train_data_group = open(output_directory+tp + "_train_group.txt", "w")

        test_data_path = output_directory+tp+"test.txt"
        test_data_store=dict()


        feature_id=1
        test_feature_id=1

        gene_list=[gene for gene in gene_set]



        # generate the partial gene list
        part_gene_set=set()

        for disease in disease_gene.keys():
            for gene in disease_gene[disease]:
                part_gene_set.add(gene)
        part_gene_list=[gene for gene in part_gene_set]

        # generate the disease set
        disease_list=[disease for disease in disease_gene.keys()]


        positive_data=dict()
        for disease in disease_gene.keys():
            positive_data[disease]=disease_gene[disease]
            for gene in disease_gene[disease]:
                try:
                    positive_data[gene].append(disease)
                except:
                    positive_data[gene]=[disease]
        num_of_negative=40*(len(part_gene_list)+len(disease_list))
        print("number of gene",str(len(part_gene_list)))
        print("number of disease",str(len(disease_list)))

        # generate the negative data
        negative_data=negative_data_sampling(part_gene_list,disease_list,num_of_negative,positive_data,model)
        print("the number of negative",str(len(negative_data)))



        training_genes=set()
        testing_genes=set()
        non_trained_genes = set()
        for disease in disease_gene.keys():


            disease_vec=model[disease]

            if disease in train_disease:
                group_number = 0
                genes=disease_gene[disease]

                # generate the positive data
                for gene in genes:
                    training_genes.add(gene)

                    gene_vec=model[gene]
                    gene_disease_vec=np.append(disease_vec,gene_vec,0)

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
                    random_index=random.choice(range(len(negative_data)))
                    if (random_index not in random_index_list):
                            random_index_list.append(random_index)

                for index in random_index_list:
                    gene_disease_vec=negative_data[index]

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
                for gene in genes:
                    testing_genes.add(gene)
                test_data_store[disease]=genes


        train_data.close()
        train_data_group.close()
        with open(test_data_path,"wb") as f:
            pkl.dump(test_data_store,f)
        for gene in testing_genes:
            if gene not in training_genes:
                non_trained_genes.add(gene)

        print(len(non_trained_genes))
        with open(output_directory+"non_train_disease.pkl","wb") as f:
            pkl.dump(non_trained_genes,f)




def negative_data_sampling(gene_set,disease_set,num_of_negative,positive_data,model):
    negative_data=[]
    while len(negative_data)<num_of_negative/2:
        gene=random.choices(gene_set)[0]
        disease=random.choices(disease_set)[0]
        if disease not in positive_data[gene]:
            gene_vec=model[gene]
            disease_vec=model[disease]
            gene_disease_vec=np.append(gene_vec,disease_vec,0)
            negative_data.append(gene_disease_vec)

    while len(negative_data)<num_of_negative:
        gene=random.choices(gene_set)[0]
        disease=random.choices(disease_set)[0]
        if gene not in positive_data[disease]:
            gene_vec=model[gene]
            disease_vec=model[disease]
            disease_gene_vec=np.append(disease_vec,gene_vec,0)
            negative_data.append(disease_gene_vec)
    random.shuffle(negative_data)

    return negative_data
generate_train_data()