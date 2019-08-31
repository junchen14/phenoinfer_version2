import sys
import xgboost as xgb
import pandas as pd
import pickle as pkl
import gensim
from scipy.stats import rankdata
import numpy as np

tp=sys.argv[1]

print(tp)
train_data="../data/boost_bidirectional_sampling_full_gene/"+tp+"_train.txt"
train_group_data="../data/boost_bidirectional_sampling_full_gene/"+tp+"_train_group.txt"


with open("../data/boost_bidirectional_sampling_full_gene/non_trained_gene.pkl","rb") as f:
    non_train_gene=pkl.load(f)

# test_data="../data/boost_data/"+tp+"_test.txt"
# test_group_data="../data/boost_data/"+tp+"_test_group.txt"

with open("../data/boost_bidirectional_sampling_full_gene/"+tp+"test.txt","rb") as f:
    disease_gene_dic=pkl.load(f)


gene_list="../data/"+tp+"_gene_set.pkl"
with open(gene_list, "rb") as f:
    gene_list = pkl.load(f)



opamodel=gensim.models.Word2Vec.load("../opamodel/"+tp+".model")

## the original rank
# index_to_rank=dict()
# index=0
# with open(test_data,"r") as f:
#     for data in f.readlines():
#         index_to_rank[index]=str(data[0])
#         index+=1

def run(objective):

    print(objective)
    dtrain = xgb.DMatrix(train_data)
    dtrain_group = load_group_file(train_group_data)
    dtrain.set_group(dtrain_group)

    # dvali = xgb.DMatrix(test_data)
    # dvali_group = load_group_file(test_group_data)
    # dvali.set_group(dvali_group)

    # dtest = xgb.DMatrix(test_data)
    # dtest_group = load_group_file(test_group_data)
    # dtest.set_group(dtest_group)


    # rank map:  eval could reach 0.7 and the train map could reach 0.81
    # rank ndcg: eval could reach 0.7 and train map could reach 0.82
    # rank pairwise eval could reach 0.2 and train map could reach 0.95


    params = {"objective": "rank:"+objective, "eta": 0.1,"gamma": 1.0,"min_child_weight": 1,"max_depth": 10}


    # watchlist = [(dvali, 'eval'), (dtrain, 'train')]
    num_round = 200


    bst=xgb.train(params,dtrain,num_round)

    pkl.dump(bst,open("xgboost_change_order.model.pt","wb"))
    # predict
    predict_performance=predict(bst,disease_gene_dic)
    print("the average rank is : ",predict_performance)





import numpy as np


def predict(model, test_disease):
    average_rank = []
    non_train_gene_rank=[]
    indd=0
    for disease in test_disease:
        indd+=1
        if indd%10==0:
            print(str(indd)+"--------------->>>>>>>>>>")
        dtest_data=write_file(disease,test_disease)

        dtest=xgb.DMatrix(dtest_data)
        dtest = xgb.DMatrix("performance/test.txt")
        dtest_group = load_group_file("performance/test_group.txt")


        dtest.set_group(dtest_group)
        pred_result=model.predict(dtest)
        gene_performance=dict()
        for gene_name,performance in zip(gene_list,pred_result):
            gene_performance[gene_name]=float(performance)

        result=sorted(gene_performance.items(), key=lambda x: x[1], reverse=True)


        genes=test_disease[disease]

        rank=0
        for data in result:
            rank+=1
            gene=data[0]
            if gene in genes:
                if gene in non_train_gene:
                    non_train_gene_rank.append(rank)
                    print(rank)
                average_rank.append(rank)

    final_result=np.mean(average_rank)

    print("the number of non train gene rank",np.mean(non_train_gene_rank))
    print(top_k(average_rank,10))
    print(top_k(average_rank,30))
    print(top_k(average_rank,50))
    print(top_k(average_rank,100))
    print(top_k(average_rank,500))

    return final_result


def top_k(result,k):
    top_k_result=[]
    for data in result:
        if data <k:
            top_k_result.append(data)
    return len(top_k_result)/len(result)


def write_file_pandas(disease,test_disease):
    data_pandas=pd.DataFrame()
    disease_vec = opamodel[disease]
    genes = test_disease[disease]

    feature_id = 1
    test_feature_id = 1

    for gene in gene_list:
        if gene in genes:
            gene_vec = opamodel[gene]
            gene_disease_vec = np.append(gene_vec, disease_vec, 0)
            tmp_data=[]

            tmp_data.append("1")
            for ind in range(len(gene_disease_vec)):
                if ind != len(gene_disease_vec) - 1:
                    tmp_data.append(str(test_feature_id) + ":" + str(gene_disease_vec[ind]))
                else:
                    tmp_data.append(str(test_feature_id) + ":" + str(gene_disease_vec[ind]))
                    test_feature_id = 0
                test_feature_id += 1
            tmp_frame=pd.DataFrame(tmp_data)
            data_pandas.append(tmp_frame)
        else:
            gene_vec = opamodel[gene]
            gene_disease_vec = np.append(gene_vec, disease_vec, 0)

            tmp_data = []

            tmp_data.append("0")
            for ind in range(len(gene_disease_vec)):
                if ind != len(gene_disease_vec) - 1:
                    tmp_data.append(str(test_feature_id) + ":" + str(gene_disease_vec[ind]))
                else:
                    tmp_data.append(str(test_feature_id) + ":" + str(gene_disease_vec[ind]))
                    test_feature_id = 0
                test_feature_id += 1
            tmp_frame = pd.DataFrame(tmp_data)
            data_pandas.append(tmp_frame)
    return data_pandas.values


def write_file(disease,test_disease):
    test_data = open("performance/test.txt", "w")
    test_group = open("performance/test_group.txt", "w")
    disease_vec = opamodel[disease]
    genes = test_disease[disease]

    feature_id = 1
    test_feature_id = 1

    for gene in gene_list:
        if gene in genes:
            gene_vec = opamodel[gene]
            gene_disease_vec = np.append(disease_vec, gene_vec, 0)

            test_data.write("1 ")
            for ind in range(len(gene_disease_vec)):
                if ind != len(gene_disease_vec) - 1:
                    test_data.write(str(test_feature_id) + ":" + str(gene_disease_vec[ind]) + " ")
                else:
                    test_data.write(str(test_feature_id) + ":" + str(gene_disease_vec[ind]) + "\n")
                    test_feature_id = 0
                test_feature_id += 1
        else:
            gene_vec = opamodel[gene]
            gene_disease_vec = np.append(disease_vec, gene_vec, 0)

            test_data.write("0 ")
            for ind in range(len(gene_disease_vec)):
                if ind != len(gene_disease_vec) - 1:
                    test_data.write(str(test_feature_id) + ":" + str(gene_disease_vec[ind]) + " ")
                else:
                    test_data.write(str(test_feature_id) + ":" + str(gene_disease_vec[ind]) + "\n")
                    test_feature_id = 0
                test_feature_id += 1

    test_group.write(str(len(gene_list)) + "\n")
    test_data.close()




def load_group_file(file_path):
    group = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                group.append(int(line.strip()))
            except Exception as ex:
                print ("Exception happen at line:", line)

    return group



# run("map")
run("pairwise")
# run("ndcg")



















def compare_list(valid_result, test_result):

    rank_list = []
    test_result=sorted(test_result,reverse=True)
    for validation in valid_result:
        for index in range(len(test_result)):
            if (validation >= test_result[index]):
                rank = rankdata(test_result, method="average")
                rank_list.append(len(gene_list) - rank[index])
                break

    if (rank_list==[]):
        print(valid_result)


    return rank_list


def compute_performance(original_rank, predicted_rank):
    print("the length of original_rank:  ",len(original_rank))
    print("the length of predicted rank:  ",len(predicted_rank))
    ranks = []
    another_index = 1
    predicted_value = []
    ranked_list = []
    print("the num of data examples",len(original_rank)/len(gene_list))
    for index in range(len(original_rank)):
        if another_index % len(gene_list) == 0:
            ranked_list.append(float(predicted_rank[index]))
            if original_rank[index] == "1":
                predicted_value.append(float(predicted_rank[index]))

            average_rank = compare_list(predicted_value, ranked_list)

            #             print(len(predicted_value),len(ranked_list))
            #             print(average_rank)
            for r in average_rank:
                ranks.append(r)
            another_index = 1
            predicted_value = []
            ranked_list = []
        else:
            ranked_list.append(float(predicted_rank[index]))
            if original_rank[index] == "1":
                predicted_value.append(float(predicted_rank[index]))

            another_index += 1

    clean_rank=[]
    for data in ranks:
        if data<100000:
            clean_rank.append(data)
        else:
            print(data)
            print("something wrong with algorithm")

    return np.mean(clean_rank)


def check_overfitting(top_rank, predicted_rank):
    top_array = []
    for index, value in enumerate(predicted_rank):
        if len(top_array) < top_rank:
            top_array.append(index)
        else:
            min_index = -100
            min_value = -1000
            for ind in range(len(top_array)):
                if predicted_rank[top_array[ind]] > min_value:
                    min_index = ind
                    min_value = predicted_rank[top_array[ind]]

            if min_index != -100:
                top_array[min_index] = index
    return sorted(top_array)


def check_ovf(original_rank, predicted_rank):
    another_index = 1
    predicted_value = []
    ranked_list = []
    for index in range(len(original_rank)):
        if another_index % len(gene_list) == 0:
            ranked_list.append(float(predicted_rank[index]))
            #             average_rank=compare_list(predicted_value,ranked_list)
            print(check_overfitting(20, ranked_list))
            #             print(len(predicted_value),len(ranked_list))
            #             print(average_rank)

            another_index = 1
            predicted_value = []
            ranked_list = []
        else:
            ranked_list.append(float(predicted_rank[index]))
            if original_rank[index] == "1":
                predicted_value.append(float(predicted_rank[index]))

            another_index += 1

