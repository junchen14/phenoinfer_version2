import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import pickle as pkl
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import random
import gensim
from sklearn import metrics
import Net
from scipy.stats import rankdata
import sys

random.seed(100)
torch.manual_seed(0)
torch.set_default_tensor_type(torch.DoubleTensor)

type=sys.argv[1]
print(type)
negative_sampling=50

word2vec_model=gensim.models.Word2Vec.load("../opamodel/"+type+".model")
with open("../data/"+type+"_positive_data.pkl","rb") as f:
    positive_data=pkl.load(f)

with open("../data/"+type+"_negative_data.pkl","rb") as f:
    negative_data=pkl.load(f)

with open("../data/"+type+"_eval_data.pkl","rb") as f:
    eval_dic=pkl.load(f)

with open("../data/"+type+"_gene_set.pkl","rb") as f:
    gene_set=pkl.load(f)



'''
now we process the data to make it suitable for training

'''

positive_numpy=np.array(positive_data, dtype="float64")
negative_numpy=np.array(negative_data, dtype="float64")
positive_train=torch.from_numpy(positive_numpy).double()
negative_train=torch.from_numpy(negative_numpy).double()


# generate the label for training data

positive_label=torch.ones(int(len(positive_data)),1).double()
negative_label=torch.zeros(int(len(negative_data)),1).double()

# concatenate the training data together
train_x=torch.cat([positive_train,negative_train],dim=0)
train_y=torch.cat([positive_label,negative_label],dim=0)



'''
now we define the neural network model and define the training method

'''


def train_epoch(model,opt,criterion):
    model.train()
    losses=0
    loader=TensorDataset(train_x,train_y)
    loader_dataset=DataLoader(loader,batch_size=125, shuffle=True)

    for x_train,y_train in loader_dataset:
        x_batch=Variable(x_train)
        y_batch=Variable(y_train)

        # predict
        y_hat=model(x_batch)

        # define the loss
        loss=criterion(y_hat,y_batch)
        opt.zero_grad()
        loss.backward()

        # update the gradient
        opt.step()
        losses+=loss.item()

    return model,losses






def test_epoch(model):
    model.eval()
    validation_rank=dict()

    for disease in eval_dic.keys():

        genes=eval_dic[disease]
        disease_vec = word2vec_model[disease]

        testing_list=[]
        for gene in gene_set:
            gene_vec=word2vec_model[gene]
            gene_disease=np.append(gene_vec,disease_vec,0)
            testing_list.append(gene_disease)

        validation_list=[]
        for gene in genes:
            gene_vec=word2vec_model[gene]
            gene_disease=np.append(gene_vec,disease_vec,0)
            validation_list.append(gene_disease)


        testing_numpy=np.array(testing_list,dtype="float64")
        data_test=torch.from_numpy(testing_numpy).double()

        validation_numpy=np.array(validation_list,dtype="float64")
        data_valid=torch.from_numpy(validation_numpy).double()

        data_test=Variable(data_test)
        data_valid=Variable(data_valid)
        test_result=sorted(model(data_test).data, reverse=True)
        valid_result=model(data_valid).data

        # calculate the mean rank for each validation result

        rank_list=[]
        for validation in valid_result:
            for index in range(len(test_result)):
                if (validation >=test_result[index]):
                    rank=rankdata(test_result,method="average")
                    rank_list.append(len(gene_set)-rank[index])
                    break

        validation_rank[disease]=rank_list


    return validation_rank


def count_h(k,rank_list):
    if (k==0):
        return 0
    threshold = k / len(gene_set)
    count = 0
    for value in rank_list:
        if value <= threshold:
            count += 1

    return count / len(rank_list)


def ranked_auc(rank_list):
    rank_dic = {}
    for i in range(1, len(gene_set)):
        rank_dic[i] = count_h(i,rank_list)
    auc = 0
    prior = 10000
    for data in rank_dic.values():
        if (prior == 10000):
            prior = data
        else:
            auc += (1 / 2) * (prior + data) / (len(gene_set) - 1)
            prior = data
    return auc


# def ranked_auc(rank_list):
#     rank_dic=dict()
#     for i in range(1, len(gene_set)):
#         rank_dic[i]=


def prediction(model):
    rank_list=[]
    rank_result=test_epoch(model)

    average_rank=0.0
    count=0
    for disease in rank_result.keys():
        for score in rank_result[disease]:
            rank= score/len(gene_set)
            rank_list.append(rank)
    auc=ranked_auc(rank_list)

    return auc



def main():
    best_performance=0
    epoch_num=201
    learning_rate=[0.01,0.005,0.001,0.0005]
    test_epoch=[10,20,35,50,100,150,200,250,300]

    for lr in learning_rate:
        model = Net.Net1().double()
        opt = optim.Adam(model.parameters(), lr=0.003, betas=(0.9, 0.999))
        criterion = nn.BCELoss()

        for g in opt.param_groups:
            g["lr"]=lr

        for epoch in range(epoch_num):
            model,losses=train_epoch(model,opt,criterion)

            if (epoch in test_epoch):
                auc=prediction(model)
                print("current epoch: ",epoch,"   current auc:  ",auc,"   best auc:  ",best_performance,
                      "  current lr:  ",lr)
                if (auc >best_performance):
                    torch.save(model,"../nnmodel/"+type+"__"+str(auc)+".pt")


main()

