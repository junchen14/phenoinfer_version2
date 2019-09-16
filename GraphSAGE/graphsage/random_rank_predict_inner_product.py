from __future__ import print_function
import json
import numpy as np
import pickle as pkl
import gensim
from networkx.readwrite import json_graph
from argparse import ArgumentParser

from random_rank_model_inner_product import Rank_model
import torch
import random
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torch.optim as optim
from scipy.stats import rankdata
import sys





''' To evaluate the embeddings, we run a logistic regression.
Run this script after running unsupervised training.
Baseline of using features-only can be run by setting data_dir as 'feat'
Example:
  python eval_scripts/ppi_eval.py ../data/ppi unsup-ppi/n2v_big_0.000010 test
'''



negative_number=20


def negative_sampling(disease_gene,gene_set,central_disease):


    positive_genes=disease_gene[central_disease]
    negative_genes=[]
    while(len(negative_genes)<negative_number):
        gene=random.choices(gene_set)
        if gene not in positive_genes:
            negative_genes.append(gene[0])

    return negative_genes


def generate_label(positive_g,negative_genes):
    label=[]

    train_gene=[]
    negative_gene=[]
    train_gene.append(positive_g)
    label.append(1)
    for gene in negative_genes:


        train_gene.append(gene)
        label.append(0)


        # if label_select==1:
        #
        #     positive_gene.append(positive_g)
        #     negative_gene.append(gene)
        # else:
        #
        #     negative_gene.append(gene)
        #     positive_gene.append(positive_g)

    return train_gene,label


def generate_embedding(embed_dic,entities):

    embed_list=[]
    for entity in entities:

        # id=id_mapping[entity]

        embed_list.append(embed_dic[entity])
    return embed_list


def convert_to_torch_format(data):
    data=np.array(data,dtype="float64")
    return torch.from_numpy(data).double()


def generate_train_data(embed_dic,disease_gene,gene_set,disease_set):
    g1 = []

    d = []
    y = []

    for disease in disease_set:
        genes = disease_gene[disease]
        for gene in genes:
            negative_genes = negative_sampling(disease_gene, gene_set, disease)
            train_gene, label = generate_label(gene, negative_genes)

            train_gene = generate_embedding(embed_dic, train_gene)


            disease_embedding = generate_embedding(embed_dic, [disease]*(negative_number+1))

            g1.extend(train_gene)

            d.extend(disease_embedding)
            y.extend(label)
    g1 = convert_to_torch_format(g1)

    d = convert_to_torch_format(d)
    y = convert_to_torch_format(y)
    return g1, d,y


def load_data(embed_dic,disease_gene,gene_set,train_diseases):

    disease_set=[key for key in disease_gene.keys()]
    #train_diseases=disease_set[:int(len(disease_genes)*0.8)]
    test_diseases=dict()
    for disease in disease_gene.keys():
        if disease in train_diseases:
            # genes=disease_gene[disease]
            # for gene in genes:
            #     negative_genes=negative_sampling(disease_gene,gene_set,disease)
            #     positive_gene,negative_gene,label=generate_label(gene,negative_genes)
            #
            #
            #     positive_gene= generate_embedding(embeddings,id_maping,positive_gene)
            #
            #     negative_gene=generate_embedding(embeddings,id_maping,negative_gene)
            #     disease_embedding=generate_embedding(embeddings,id_maping,[disease]*negative_number)
            #
            #     g1.extend(positive_gene)
            #     g2.extend(negative_gene)
            #     d.extend(disease_embedding)
            #     y.extend(label)
            pass

        else:
            test_diseases[disease]=disease_gene[disease]
    g1,d,y=generate_train_data(embed_dic,disease_gene,gene_set,train_diseases)




    return g1,d,y,test_diseases,train_diseases


def train(model_,opt_,criterion_):
    model_.train()
    loader = TensorDataset(g1, d, y)
    loader_dataset = DataLoader(loader, batch_size=125, shuffle=True)
    losses = 0

    for gene1_,disease_, y_label in loader_dataset:
        gene1_batch = Variable(gene1_)

        disease_batch = Variable(disease_)

        predict_y = model_(gene1_batch,disease_batch)
        loss = criterion_(predict_y, y_label)

        opt_.zero_grad()
        loss.backward()
        opt_.step()
        losses += loss.item()

    return model, losses

def look_up_embed(embed_dic_,entity):
    # entity_emb=id_map_[entity]
    # entity_emb=embedding_[entity_emb]
    entity_emb=embed_dic_[entity]

    return entity_emb

def top_k(evaluation,k):
    scores=[]
    for key in evaluation.keys():
        scores.extend(evaluation[key])
    # print(evaluation)
    # print("---------")
    # print(scores)
    top_k_result=[]
    for data in scores:
        if data <k:
            top_k_result.append(data)
    rank=len(top_k_result)/len(scores)
    print("the top "+str(k)+" rank is :",str(rank))



def evaluation(model_,evaluation_data,embed_dic_,gene_set_):
    model_.eval()
    validation_rank = dict()
    for disease in evaluation_data.keys():
        positive_genes=evaluation_data[disease]

        disease_vec=look_up_embed(embed_dic_,disease)


        testing_list = []
        for gene in gene_set_:
            gene_vec = look_up_embed(embed_dic_,gene)
            testing_list.append([gene_vec,disease_vec])

        validation_list = []
        for gene in positive_genes:
            gene_vec = look_up_embed(embed_dic_,gene)
            validation_list.append([gene_vec,disease_vec])

        testing_list=convert_to_torch_format(testing_list)
        validation_list=convert_to_torch_format(validation_list)
        testing_list=Variable(testing_list)
        validation_list=Variable(validation_list)

        test_result=sorted(model_.predict(testing_list).data,reverse=True)
        valid_result=model_.predict(validation_list).data

        rank_list = []
        # print("test result")
        # print(test_result)

        for validation in valid_result:

            for index in range(len(test_result)):

                if (validation >= test_result[index]):
                    rank = rankdata(test_result, method="average")
                    rank_list.append(len(gene_set_) - rank[index])
                    break

        validation_rank[disease] = rank_list


    top_k(validation_rank,10)
    top_k(validation_rank,30)
    top_k(validation_rank,50)
    top_k(validation_rank,100)
    top_k(validation_rank,200)

    average_score=[]
    for disease in validation_rank.keys():
        average_score.extend(validation_rank[disease])
    mean_rank=np.mean(average_score)

    print("the mean rank of the whole data is ",str(mean_rank))
    return validation_rank,mean_rank




def count_h(k,rank_list):
    if (k==0):
        return 0
    threshold = k / len(gene_list)
    count = 0
    for value in rank_list:
        if value <= threshold:
            count += 1

    return count / len(rank_list)


def ranked_auc(rank_list):
    rank_dic = {}
    for i in range(1, len(gene_list)):
        rank_dic[i] = count_h(i,rank_list)
    auc = 0
    prior = 10000
    for data in rank_dic.values():
        if (prior == 10000):
            prior = data
        else:
            auc += (1 / 2) * (prior + data) / (len(gene_list) - 1)
            prior = data
    return auc


def calculate_auc(rank_result):
    rank_list = []

    for disease in rank_result.keys():
        for score in rank_result[disease]:
            rank= score/len(gene_list)
            rank_list.append(rank)
    auc=ranked_auc(rank_list)
    print("the auc is :",str(auc))







if __name__ == '__main__':
    # parser = ArgumentParser("Run evaluation on PPI data.")
    # parser.add_argument("data_type", help="Path to directory containing the dataset.")
    # parser.add_argument("embed_dir",
    #                     help="Path to directory containing the learned node embeddings. Set to 'feat' for raw features.")
    # # parser.add_argument("setting", help="Either val or test.")
    # args = parser.parse_args()
    #
    #
    # data_type = args.data_type
    # data_dir = args.embed_dir

    data_type=sys.argv[1]
    #data_dir="../unsup-gene_disease/graphsage_mean_small_0.000010"


    # setting = args.setting
    disease_gene_path="../data/"+data_type+"_disease_gene.pkl"
    gene_set_path="../data/"+data_type+"_gene_set.pkl"



    with open(disease_gene_path,"rb") as f:
        disease_genes=pkl.load(f)

    with open(gene_set_path,"rb") as f:
        gene_sets=pkl.load(f)
    gene_list=[gene for gene in gene_sets]




    # i = 0
    # dic = dict()
    # import numpy as np
    #
    # with open("../"+data_type+"_graph/walks-vec.txt", "r") as f:
    #     for line in f.readlines():
    #         if i != 0:
    #             data = line.strip().split()
    #             entity = data[0]
    #             embedding = data[1:]
    #             emb = np.asarray(embedding)
    #             dic[entity] = emb
    #         else:
    #             i += 1

    with open("../data/"+data_type+"_disease_gene.pkl","rb") as f:
        disease_gene=pkl.load(f)
    entities=set()
    for disease in disease_gene.keys():
        entities.add(disease)
        for gene in disease_gene[disease]:
            entities.add(gene)
    for gene in gene_list:
        entities.add(gene)

    dic=dict()
    word2vec_model=gensim.models.Word2Vec.load("../small_graph"+data_type+"-model_word2vec")
    for entity in entities:
        dic[entity]=word2vec_model[entity]
    print("already loaded the data")
    print(len(dic))





    with open("../small_graph/"+data_type+"_train_disease.pkl","rb") as f:
        train_disease=pkl.load(f)


    # with open("../test_graph/test_disease.pkl") as f:
    #     test_disease=pkl.load(f)


    print("loading data....")
    G=json_graph.node_link_graph(json.load(open("../small_graph/"+data_type+"-gd-G.json")))
    train_ids=[n for n in G.nodes() if not G.node[n]["val"]]


    g1,d,y,test_disease,train_disease   =  load_data(dic,disease_genes,gene_list,train_disease)

    print("g1",g1.shape)

    print("d",d.shape)
    print("y",y.shape)

    feature_num=g1.shape[1]
    print("the feature number is",feature_num)
    model=Rank_model(num_feature=feature_num).double()
    opt = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    criterion = torch.nn.BCELoss()

    epoches=1000

    performance=1000000
    for epoch in range(epoches):
        g1, d, y= generate_train_data(dic,disease_genes,gene_list,train_disease)
        if epoch%100==5:
            auc,rank=evaluation(model,test_disease,dic,gene_list)
            calculate_auc(auc)
            if rank<performance:
                torch.save(model, "../model/"+data_type+"__" + str(rank) + ".pt")

        model,loss_value=train(model,opt,criterion)
        print("the loss is:  ",str(loss_value))
