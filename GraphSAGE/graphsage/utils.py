from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph
import multiprocessing as mp
from threading import Lock
import gensim


lock = Lock()

version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
# assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN=10
N_WALKS=50

global data_pairs
data_pairs = []


def load_data(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(list(G.nodes())[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    # class_map = json.load(open(prefix + "-class_map.json"))
    # if isinstance(list(class_map.values())[0], list):
    #     lab_conversion = lambda n : n
    # else:
    #     lab_conversion = lambda n : int(n)
    #
    # class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    # ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'val' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['val'] or G.node[edge[1]]['val']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        # train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['val']])
        train_ids=np.array([id_map[n] for n in G.nodes()])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    if load_walks:
        with open(prefix + "-walks.txt","r") as fp:
            for line in fp:
                temp_data=[line.split()[0].strip(),line.split()[0].strip()]
                walks.append(temp_data)

                # walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks

def run_random_walks(G, nodes, num_walks=N_WALKS):

    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(list(G.neighbors(curr_node)))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")



    # data_pairs.extend(pairs)

    write_file(pairs)
    # return pairs

def run_walk(nodes,G):
    global data_pairs

    number=20
    length = len(nodes) // number

    processes = [mp.Process(target=run_random_walks, args=(G, nodes[(index) * length:(index + 1) * length])) for index
                 in range(number-1)]
    processes.append(mp.Process(target=run_random_walks, args=(G, nodes[(number-1) * length:len(nodes) - 1])))

    for p in processes:
        p.start()
    for p in processes:
        p.join()



    print("finish the work here")

def write_file(pair):
    with lock:
        with open(out_file, "a") as fp:
            fp.write("\n".join([str(p[0]) + " " + str(p[1]) for p in pair]))
            fp.write("\n")


if __name__ == "__main__":
    """ Run random walks """

    root_directory = sys.argv[1]
    data_type=sys.argv[2]

    graph_file = root_directory+"/"+data_type+"-gd-G.txt"
    graph_walks = root_directory+"/"+data_type+"-gd-walks.txt"
    graph_vectors= root_directory+"/"+data_type+"-model_word2vec"



    file=open(graph_walks,"w")
    file.close()
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    # nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]

    nodes= [n for n in G.nodes()]
    G = G.subgraph(nodes)
    run_walk(nodes,G)

    print("start to train the word2vec models")
    sentences=gensim.models.word2vec.LineSentence(graph_walks)
    model=gensim.models.Word2Vec(sentences,sg=1, min_count=1, size=100, window=3,iter=30,workers=20)
    model.save(graph_vectors)
