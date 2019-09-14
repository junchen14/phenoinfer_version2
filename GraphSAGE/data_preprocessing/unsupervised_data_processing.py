import networkx as nx
import pickle as pkl
import json
import sys

data_type=sys.argv[1]

axiom_file="data/axioms.lst"
annotation_association="GraphSAGE/data/"+data_type+"_train.txt"
disease_gene_association="GraphSAGE/data/"+data_type+"_disease_gene.pkl"

entity_list=set()

def generate_eneity_list():
    with open(annotation_association, "r") as f:
        for line in f.readlines():
            entities = line.split()
            entity_list.add(entities[0].strip())
            entity_list.add(entities[1].strip())

    with open(disease_gene_association, "rb") as f:
        disease_gene = pkl.load(f)
    for disease in disease_gene.keys():
        entity_list.add(disease)
        for value in disease_gene[disease]:
            entity_list.add(value)




generate_eneity_list()


def generate_graph(association1, association2):
    G = nx.Graph()
    non_match_symbol = ["", "and", "some"]
    with open(axiom_file, "r") as f:
        for line in f.readlines():
            entities = line.split(" ")
            if len(entities) < 3:
                print(entities)
            else:
                first_entity = entities[0].strip()
                if first_entity in entity_list:
                    edge_type = entities[1].strip()
                    for i in range(2, len(entities)):
                        if entities[i].strip() not in non_match_symbol:
                            G.add_edge(first_entity, entities[i].strip())
                            G.edges[first_entity, entities[i].strip()]["type"] = edge_type
                            G.nodes[first_entity]["val"] = False
                            G.nodes[entities[i].strip()]["val"] = False
                else:
                    edge_type = entities[1].strip()
                    for i in range(2, len(entities)):
                        if entities[i].strip() not in non_match_symbol:
                            if entities[i].strip() in entity_list:
                                G.add_edge(first_entity, entities[i].strip())
                                G.edges[first_entity, entities[i].strip()]["type"] = edge_type
                                G.nodes[first_entity]["val"] = False
                                G.nodes[entities[i].strip()]["val"] = False

                        # G[first_entity][entities[i].strip()]=edge_type
    with open(association1, "r") as f:
        for line in f.readlines():
            entities = line.split()
            G.add_edge(entities[0].strip(), entities[1].strip())
            G.edges[entities[0].strip(), entities[1].strip()]["type"] = "EquivalentTo"
            G.nodes[entities[0].strip()]["val"] = False
            G.nodes[entities[1].strip()]["val"] = False

    with open(association2, "rb") as f:
        disease_gene = pkl.load(f)
    diseases = [disease_gene.keys()]
    import random
    random.shuffle(diseases)
    train_disease = diseases[:int(len(diseases) * 0.8)]
    # new_train_disease = []
    # for disease in train_disease:
    #     if disease not in test_disease.keys():
    #         new_train_disease.append(disease)

    for key in disease_gene.keys():
        if key in train_disease:
            for gene in disease_gene[key]:
                G.add_edge(str(key), str(gene))
                G.edges[str(key), str(gene)]["type"] = "EquivalentTo"
                G.nodes[key]["val"] = False
                G.nodes[gene]["val"] = False


    return G


G = generate_graph(annotation_association, disease_gene_association)
from networkx.readwrite import json_graph

graph1 = json_graph.node_link_data(G)
with open("../small_graph/"+data_type"-gd-G.json", "w") as f:
    json.dump(graph1, f)

dic = dict()
index = 0

for node in G.nodes():
    dic[node] = index
    index += 1

print('the numberof ndoes in the graph', str(index))
