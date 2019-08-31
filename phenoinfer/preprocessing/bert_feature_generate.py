from bert_serving.client import BertClient
bc = BertClient()


import pandas as pd
import pickle as pkl


options=["intersection","go","uberon","mgi","union"]

# generate the human disease and hp association
dis_phe=dict()
with open("../../phenotype_annotation.tab","r") as f:
    for line in f.readlines():
        data=line.split("\t")
        if (data[0]=="OMIM")&(data[5][:4]=="OMIM"):
            try:
                dis_phe[data[5].strip()].append(data[4].strip())
            except:
                dis_phe[data[5].strip()]=[data[4].strip()]

print("the number of disease phenotypes ",len(dis_phe))


# obtain the mouse gene and mouse disease association
# obtain the human gene and human disease association


mgi_do=pd.read_table("../../MGI_DO.rpt.txt")
human_gene_disease=[]

human_name_data=pd.read_table("../../gene_name.txt")
id_gene_name=dict()
from gensim.parsing.preprocessing import remove_stopwords
for index in human_name_data.index:

    name=human_name_data.loc[index,"name"]
    id=human_name_data.loc[index,"entrez_id"]
    if str(name)!="nan":
        if str(id)!="nan":
            print(name)
            result=precessed_name=remove_stopwords(name)
            print(result)
            id=str(int(id))
            id_gene_name[id]=name


id_to_gene_name=dict()
ind=0
with open("../../HMD_HumanPhenotype.rpt.txt",'r') as f:
    for line in f.readlines():
        data=line.split("\t")
        gene_name=data[0].strip()
        human_id=str(int(data[1].strip()))
        if human_id in id_gene_name.keys():
            embedding=bc.encode([id_gene_name[id]])
            print(len(embedding[0]))
            id_to_gene_name[human_id]=embedding[0]
        else:
            ind+=1

with open("../data/gene_name_embedding.pkl","wb") as f:
    pkl.dump(id_to_gene_name,f)


print(len(id_to_gene_name))
print("the num of gene that does not exist:  ",len(ind))



