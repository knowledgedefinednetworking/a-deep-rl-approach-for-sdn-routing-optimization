"""
Traffic.py
"""
__author__ = "giorgio@ac.upc.edu"

import numpy as np
from os import listdir
from re import split

from OU import OU
from helper import softmax


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in split(r'(\d+)', string_)]


class Traffic():

    def __init__(self, nodes_num, type, capacity):
        self.nodes_num = nodes_num
        self.prev_traffic = None
        self.type = type
        self.capacity = capacity * nodes_num / (nodes_num - 1)
        self.dictionary = {}
        self.dictionary['NORM'] = self.normal_traffic
        self.dictionary['UNI'] = self.uniform_traffic
        self.dictionary['CONTROLLED'] = self.controlled_uniform_traffic
        self.dictionary['EXP'] = self.exp_traffic
        self.dictionary['OU'] = self.ou_traffic
        self.dictionary['STAT'] = self.stat_traffic
        self.dictionary['STATEQ'] = self.stat_eq_traffic
        self.dictionary['FILE'] = self.file_traffic
        self.dictionary['DIR'] = self.dir_traffic
        if self.type.startswith('DIR:'):
            self.dir = sorted(listdir(self.type.split('DIR:')[-1]), key=lambda x: natural_key((x)))
        self.static = None
        self.total_ou = OU(1, self.capacity/2, 0.1, self.capacity/2)
        self.nodes_ou = OU(self.nodes_num**2, 1, 0.1, 1)

    def normal_traffic(self):
        t = np.random.normal(capacity/2, capacity/2)
        return np.asarray(t * softmax(np.random.randn(self.nodes_num, self.nodes_num))).clip(min=0.001)

    def uniform_traffic(self):
        t = np.random.uniform(0, self.capacity*1.25)
        return np.asarray(t * softmax(np.random.uniform(0, 1, size=[self.nodes_num]*2))).clip(min=0.001)

    def controlled_uniform_traffic(self):
        t = np.random.uniform(0, self.capacity*1.25)
        if self.prev_traffic is None:
            self.prev_traffic = np.asarray(t * softmax(np.random.uniform(0, 1, size=[self.nodes_num]*2))).clip(min=0.001)
        dist = [1]
        dist += [0]*(self.nodes_num**2 - 1)
        ch = np.random.choice(dist, [self.nodes_num]*2)

        tt = np.multiply(self.prev_traffic, 1 - ch)

        nt = np.asarray(t * softmax(np.random.uniform(0, 1, size=[self.nodes_num]*2))).clip(min=0.001)
        nt = np.multiply(nt, ch)

        self.prev_traffic = tt + nt

        return self.prev_traffic

    def exp_traffic(self):
        a = np.random.exponential(size=self.nodes_num)
        b = np.random.exponential(size=self.nodes_num)

        T = np.outer(a, b)

        np.fill_diagonal(T, -1)

        T[T!=-1] = np.asarray(np.random.exponential()*T[T!=-1]/np.average(T[T!=-1])).clip(min=0.001)

        return T

    def stat_traffic(self):
        if self.static is None:
            string = self.type.split('STAT:')[-1]
            v = np.asarray(tuple(float(x) for x in string.split(',')[:self.nodes_num**2]))
            M = np.split(v, self.nodes_num)
            self.static = np.vstack(M)
        return self.static

    def stat_eq_traffic(self):
        if self.static is None:
            value = float(self.type.split('STATEQ:')[-1])
            self.static = np.full([self.nodes_num]*2, value, dtype=float)
        return self.static

    def ou_traffic(self):
        t = self.total_ou.evolve()[0]
        nt = t * softmax(self.nodes_ou.evolve())
        i = np.split(nt, self.nodes_num)
        return np.vstack(i).clip(min=0.001)

    def file_traffic(self):
        if self.static is None:
            fname = 'traffic/' + self.type.split('FILE:')[-1]
            v = np.loadtxt(fname, delimiter=',')
            self.static = np.split(v, self.nodes_num)
        return self.static

    def dir_traffic(self):
        while len(self.dir) > 0:
            tm = self.dir.pop(0)
            if not tm.endswith('.txt'):
                continue
            fname = self.type.split('DIR:')[-1] + '/' + tm
            v = np.loadtxt(fname, delimiter=',')
            return np.split(v, self.nodes_num)
        return False


    def generate(self):
        return self.dictionary[self.type.split(":")[0]]()
