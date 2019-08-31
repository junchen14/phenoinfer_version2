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
        with open("../data/vae/"+tp+"train_disease.pkl","rb") as f:
            train_disease=pkl.load(f)

        with open("../data/vae/neural_data.pkl","rb") as f:
            neutral_data=pkl.load(f)

        neutral_data=np.asarray(neutral_data)
        neutral_data=neutral_data.reshape(neutral_data.shape[0],-1)
        print(neutral_data.shape)

        model=gensim.models.Word2Vec.load("../opamodel/"+tp+"_association_only.model")


        train_data = open(output_directory+tp+"_train.txt", "w")
        train_data_group = open(output_directory+tp + "_train_group.txt", "w")

        test_data_path = output_directory+tp+"test.txt"
        test_data_store=dict()


        feature_id=1
        test_feature_id=1

        gene_list=[gene for gene in gene_set]
        random.shuffle(gene_list)


        # generate the partial gene list


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
        num_of_negative=20*(len(gene_list)+len(disease_list))
        print("number of gene",str(len(gene_list)))
        print("number of disease",str(len(disease_list)))

        # generate the negative data
        negative_data=negative_data_sampling(gene_list,disease_list,num_of_negative,positive_data,model)
        print("the number of negative",str(len(negative_data)))

        training_gene=set()
        testing_gene=set()
        non_train_gene=set()


        for disease in disease_gene.keys():


            disease_vec=model[disease]

            if disease in train_disease:
                group_number = 0
                genes=disease_gene[disease]

                # generate the positive data
                for gene in genes:
                    training_gene.add(gene)


                    gene_vec=model[gene]
                    gene_disease_vec=np.append(disease_vec,gene_vec,0)

                    group_number+=1
                    train_data.write("2 ")
                    for ind in range(len(gene_disease_vec)):
                        if ind != len(gene_disease_vec) - 1:
                            train_data.write(str(feature_id) + ":" + str(gene_disease_vec[ind]) + " ")
                        else:
                            train_data.write(str(feature_id) + ":" + str(gene_disease_vec[ind]) + "\n")
                            feature_id = 0
                        feature_id += 1


                # generate the neutral genes
                neutral_data_vae = find_nerual_genes(neutral_data, 20)
                for data in neutral_data_vae:

                    train_data.write("1 ")
                    group_number += 1
                    if ind != len(data) - 1:
                        train_data.write(str(feature_id) + ":" + str(data[ind]) + " ")
                    else:
                        train_data.write(str(feature_id) + ":" + str(data[ind]) + "\n")
                        feature_id = 0
                    feature_id += 1

                random_index_list=[]
                while(len(random_index_list)<20):
                    random_index=random.choice(range(len(negative_data)))
                    if (random_index not in random_index_list):
                            random_index_list.append(random_index)

                # generate the negative data

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
                    testing_gene.add(gene)



                test_data_store[disease]=genes


        train_data.close()
        train_data_group.close()
        with open(test_data_path,"wb") as f:
            pkl.dump(test_data_store,f)

        for gene in testing_gene:
            if gene not in training_gene:
                non_train_gene.add(gene)
        with open(output_directory+"non_trained_gene.pkl","wb") as f:
            pkl.dump(non_train_gene,f)
        print("the length of non train genes",len(non_train_gene))


def find_nerual_genes(array,number):
    index=np.random.randint(0,len(array),number)
    return array[index]

def negative_data_sampling(gene_set,disease_set,num_of_negative,positive_data,model):
    negative_data=[]
    while len(negative_data)<num_of_negative/2:
        gene=random.choices(gene_set)[0]
        disease=random.choices(disease_set)[0]
        if gene in positive_data.keys():
            if disease not in positive_data[gene]:
                gene_vec=model[gene]
                disease_vec=model[disease]
                gene_disease_vec=np.append(gene_vec,disease_vec,0)
                negative_data.append(gene_disease_vec)
        else:
            gene_vec = model[gene]
            disease_vec = model[disease]
            gene_disease_vec = np.append(gene_vec, disease_vec, 0)
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