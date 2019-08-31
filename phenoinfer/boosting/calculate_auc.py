import pickle as pkl

tp="union"
with open("../boosting/performance/result.pkl","rb") as f:
    result=pkl.load(f)

gene_list="../data/"+tp+"_gene_set.pkl"
with open(gene_list, "rb") as f:
    gene_list = pkl.load(f)

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

new_result=[value/len(gene_list) for value in result]
print(ranked_auc(new_result))