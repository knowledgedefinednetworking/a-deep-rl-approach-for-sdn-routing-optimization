"""
path.py
"""
__author__ = "giorgio@ac.upc.edu"

import numpy as np
from itertools import product
import sys
import networkx as nx


G = nx.Graph(np.loadtxt(sys.argv[1], dtype=int))

n_edges = len(G.edges())

num = 1e5
good = set()

i = 0
while len(good) < num:
    i += 1
    matrix = np.full([len(G.nodes())]*2, -1, dtype=int)

    for u, v, d in G.edges(data=True):
        d['weight'] = np.random.random()
    all_shortest = nx.all_pairs_dijkstra_path(G)

    for s, d in product(G.nodes(), G.nodes()):
        if s != d:
            matrix[s][d] = all_shortest[s][d][1]

    candidate = ','.join(map(str, matrix.ravel()))

    l = len(good)
    good.add(candidate)

    if len(good) == l:
        print('dup!')
        continue

    print(len(good), i)
    with open('path_'+ str(sys.argv[1]) +'.csv', 'a') as file:
        file.write(candidate + '\n')
