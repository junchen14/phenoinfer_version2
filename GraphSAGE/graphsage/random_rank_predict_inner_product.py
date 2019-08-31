from __future__ import print_function
import json
import numpy as np
import pickle as pkl

from networkx.readwrite import json_graph
from argparse import ArgumentParser

from graphsage.random_rank_model_inner_product import Rank_model
import torch
import random
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torch.optim as optim
from scipy.stats import rankdata
from numba import jit

''' To evaluate the embeddings, we run a logistic regression.
Run this script after running unsupervised training.
Baseline of using features-only can be run by setting data_dir as 'feat'
Example:
  python eval_scripts/ppi_eval.py ../data/ppi unsup-ppi/n2v_big_0.000010 test
'''



negative_number=20
# def run_regression(train_embeds, train_labels, test_embeds, test_labels):
#     np.random.seed(1)
#     from sklearn.linear_model import SGDClassifier
#     from sklearn.dummy import DummyClassifier
#     from sklearn.metrics import f1_score
#     from sklearn.multioutput import MultiOutputClassifier
#     dummy = MultiOutputClassifier(DummyClassifier())
#     dummy.fit(train_embeds, train_labels)
#     log = MultiOutputClassifier(SGDClassifier(loss="log"), n_jobs=10)
#     log.fit(train_embeds, train_labels)
#
#     f1 = 0
#     for i in range(test_labels.shape[1]):
#         print("F1 score", f1_score(test_labels[:, i], log.predict(test_embeds)[:, i], average="micro"))
#     for i in range(test_labels.shape[1]):
#         print("Random baseline F1 score",
#               f1_score(test_labels[:, i], dummy.predict(test_embeds)[:, i], average="micro"))


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


def load_data(embed_dic,disease_gene,gene_set):

    disease_set=[key for key in disease_gene.keys()]
    train_diseases=disease_set[:int(len(disease_genes)*0.8)]
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

    data_type="union"
    #data_dir="../unsup-gene_disease/graphsage_mean_small_0.000010"


    # setting = args.setting
    disease_gene_path="../data/"+data_type+"_disease_gene.pkl"
    gene_set_path="../data/"+data_type+"_gene_set.pkl"



    with open(disease_gene_path,"rb") as f:
        disease_genes=pkl.load(f)

    with open(gene_set_path,"rb") as f:
        gene_sets=pkl.load(f)
    gene_list=[gene for gene in gene_sets]

    i = 0
    dic = dict()
    import numpy as np

    with open("../small_graph/walks-vec.txt", "r") as f:
        for line in f.readlines():
            if i != 0:
                data = line.strip().split()
                entity = data[0]
                embedding = data[1:]
                emb = np.asarray(embedding)
                dic[entity] = emb
            else:
                i += 1


    print("loading data....")
    G=json_graph.node_link_graph(json.load(open("../small_graph/gd-G.json")))
    train_ids=[n for n in G.nodes() if not G.node[n]["val"]]


    g1,d,y,test_disease,train_disease=load_data(dic,disease_genes,gene_list)

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
                torch.save(model, "../model/"+"inner__" + str(rank) + ".pt")

        model,loss_value=train(model,opt,criterion)
        print("the loss is:  ",str(loss_value))















    # train_embeds = embeds[[id_map[id] for id in train_ids]]








    # print("Loading data...")
    # G = json_graph.node_link_graph(json.load(open(dataset_dir + "/ppi-G.json")))
    # labels = json.load(open(dataset_dir + "/ppi-class_map.json"))
    # labels = {int(i): l for i, l in labels.iteritems()}
    #
    # train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    # test_ids = [n for n in G.nodes() if G.node[n][setting]]
    # train_labels = np.array([labels[i] for i in train_ids])
    # if train_labels.ndim == 1:
    #     train_labels = np.expand_dims(train_labels, 1)
    # test_labels = np.array([labels[i] for i in test_ids])
    # print("running", data_dir)
    #
    # if data_dir == "feat":
    #     print("Using only features..")
    #     feats = np.load(dataset_dir + "/ppi-feats.npy")
    #     ## Logistic gets thrown off by big counts, so log transform num comments and score
    #     feats[:, 0] = np.log(feats[:, 0] + 1.0)
    #     feats[:, 1] = np.log(feats[:, 1] - min(np.min(feats[:, 1]), -1))
    #     feat_id_map = json.load(open(dataset_dir + "/ppi-id_map.json"))
    #     feat_id_map = {int(id): val for id, val in feat_id_map.iteritems()}
    #     train_feats = feats[[feat_id_map[id] for id in train_ids]]
    #     test_feats = feats[[feat_id_map[id] for id in test_ids]]
    #     print("Running regression..")
    #     from sklearn.preprocessing import StandardScaler
    #
    #     scaler = StandardScaler()
    #     scaler.fit(train_feats)
    #     train_feats = scaler.transform(train_feats)
    #     test_feats = scaler.transform(test_feats)
    #     run_regression(train_feats, train_labels, test_feats, test_labels)
    # else:
    #     embeds = np.load(data_dir + "/val.npy")
    #     id_map = {}
    #     with open(data_dir + "/val.txt") as fp:
    #         for i, line in enumerate(fp):
    #             id_map[int(line.strip())] = i
    #     train_embeds = embeds[[id_map[id] for id in train_ids]]
    #     test_embeds = embeds[[id_map[id] for id in test_ids]]
    #
    #     print("Running regression..")
    #     run_regression(train_embeds, train_labels, test_embeds, test_labels)
