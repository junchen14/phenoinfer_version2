import gensim
import xgboost as xgb
import pickle as pkl
import numpy as np

type=["1","2","3","4","5","6"]
for tp in type:
    file_to_test="test"+tp+".txt"
    disease="pd"+tp
    opamodel=gensim.models.Word2Vec.load("../opamodel/union_association_only.model")
    # print(opamodel["<http://purl.obolibrary.org/obo/MP:0003743>"])
    # print(opamodel["<http://purl.obolibrary.org/obo/MP_0003743>"])

    sentences=gensim.models.word2vec.LineSentence(file_to_test)

    opamodel.build_vocab(sentences,update=True)
    print(opamodel.corpus_count)

    opamodel.train(sentences,total_examples=opamodel.corpus_count,epochs=30)


    gene_list="../data/union_gene_set.pkl"
    with open(gene_list, "rb") as f:
        gene_list = pkl.load(f)


    # with open("../data/gene_name_embedding.pkl","rb") as f:
    #     gene_name_embedding=pkl.load(f)
    gene_list=[gene for gene in gene_list]

    with open("../boosting/xgboost_vae_neutral_predict.model.pt","rb") as f:
        model=pkl.load(f)

    disease_vec=opamodel[disease]

    def write_file(test_disease):
        test_data = open("tmp/test.txt", "w")
        test_group = open("tmp/test_group.txt", "w")


        feature_id = 1
        test_feature_id = 1


        for gene in gene_list:
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


        test_group.write(str(len(gene_list)) + "\n")
        test_data.close()

    def predict(model, test_disease):



        dtest_data=write_file(test_disease)

        dtest=xgb.DMatrix(dtest_data)
        dtest = xgb.DMatrix("tmp/test.txt")
        dtest_group = load_group_file("tmp/test_group.txt")


        dtest.set_group(dtest_group)
        pred_result=model.predict(dtest)
        file=open("tmp/"+disease+".txt","w")
        gene_performance = dict()
        for gene_name, performance in zip(gene_list, pred_result):
            gene_performance[gene_name] = float(performance)
        result = sorted(gene_performance.items(), key=lambda x: x[1], reverse=True)
        for data in result:
            file.write(str(data[0])+" "+str([data[1]])+"\n")

    def load_group_file(file_path):
        group = []
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    group.append(int(line.strip()))
                except Exception as ex:
                    print ("Exception happen at line:", line)

        return group

    predict(model,disease)
