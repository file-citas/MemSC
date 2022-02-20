import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pickle
import re
import bz2
import operator
from operator import itemgetter


class GetSCs(object):
    def __init__(self, data_ma_path, data_in_path, mdl_path, n=100, typ="pr"):
        self.data_ma_path = data_ma_path
        self.data_in_path = data_in_path
        self.mdl_path = mdl_path
        self.n = n
        self.typ = typ
        self.maf = MaFeatures.MaFeatures()
        self.maf.add_data(data_ma_path, data_in_path, annot_path=None, typ=typ)
        self.maei = MaFeatures.MaEvalIter(maf.df, n)
        self.loaded_model = pickle.load(open(mdl, 'rb'))

    def eval(self):
        res = {}
        while True:
            idx = self.maei.index
            x = np.empty((0,99), dtype='float64')
            done=False
            for i in range(0, 4096):
                try:
                    x = np.append(x, [self.maei.next()], axis=0)
                except:
                    done=True
                    break
            if done:
                print("DONE")
                break
            y = self.loaded_model.predict(x.astype(np.uint64))
            for i, tmp in enumerate(y):
                if tmp != 65535:
                    print("\n%8d: %x" % (idx+i, tmp))
                    if tmp not in res.keys():
                        res[tmp] = set()
                    res[tmp].add(idx+1)

        return res



