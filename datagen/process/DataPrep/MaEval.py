import os
import json
import random
import pickle
import pandas as pd
import numpy as np
import itertools
import tempfile
import configparser
import subprocess
import sys
import multiprocessing
import json

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import scipy.stats as stats
from scipy.stats import norm

import pycaret as pc
import pycaret.classification
import sklearn.metrics as metrics

from . import MaFeatures

MAX_SIZE = 1024 * 1024 * 1024 * 16
IDX_DEC = 0
NCOLS = 2
DATA_SIZE = 4

class MaEval(object):
    def __init__(self, data_path, anot_path, info_path, scmap):
        self.scmap = scmap
        self.maf = MaFeatures.MaFeatures()
        self.maf.add_data(data_path, info_path, anot_path)

    def do_predict(self, n_feat, blur, model_base_path, step=50000, prob_y_th=0.4, prob_nc_th=0.2, verbose=False):
        print("Params: nfeat %d, blur %d" % (n_feat, blur))
        ncnr_path = "%s.NR2CNR.json" % model_base_path
        sc2key_str = json.loads(open(ncnr_path, "r").read())
        sc2key = {int(v): int(k) for v, k in sc2key_str.items()}
        key2sc = {int(v): int(k) for k, v in sc2key.items()}
        model_conf_path = "%s.py_caret_setup_config.pkl" % model_base_path
        model_path = "%s.final.model" % model_base_path
        print("Loading Config %s" % model_conf_path)
        pc.classification.load_config(model_conf_path)
        print("Loading Model %s" % model_path)
        pmdl = pc.classification.load_model(model_path)

        print("Predicting ...")
        mai = MaFeatures.MaEvalIter2(
                self.maf, step, blur, n_feat, False, drop_irq=False, rename_cols=None, sc_trans=None)
        all_sc = set()
        sc_count_pred = {}
        sc_count_base = {}
        y_base = []
        y_pred = []
        for features in mai:
            sys.stdout.write("%8d / %8d\\r" % (mai.getIndex(), mai.getTotal()))
            pred = pmdl.predict(X=features)
            #prob = pmdl.predict_proba(X=features)
            #pred2 = []
            #for i in range(len(prob)):
            #    pred_y = pred[i]
            #    prob_y = prob[i][pred_y]
            #    prob_nc = prob[i][sc2key[MaFeatures.MaFeatures.NOCLASS]]
            #    if prob_y <= prob_y_th and prob_nc >= prob_nc_th and key2sc[pred_y] != MaFeatures.MaFeatures.NOCLASS:
            #        if verbose:
            #            print("Warn %5d %5.3f %5.3f %s" % (key2sc[pred_y], prob_y, prob_nc, self.scmap[key2sc[pred_y]]))
            #        pred2.append(sc2key[MaFeatures.MaFeatures.NOCLASS])
            #    else:
            #        pred2.append(pred_y)
            #pred = pred2
            y_pred.extend(pred)
            (unique, counts) = np.unique(pred, return_counts=True)
            for i in range(len(unique)):
                sc = key2sc[unique[i]]
                if sc not in sc_count_pred.keys():
                    sc_count_pred[sc] = 0
                sc_count_pred[sc] += counts[i]
                all_sc.add(sc)
            y_base.extend([sc2key[t] for t in features['KEY'].values])
            for pdk, pdg in features.groupby(['KEY']):
                sc = int(pdk)
                if sc not in sc_count_base.keys():
                    sc_count_base[sc] = 0
                sc_count_base[sc] += len(pdg.index)
                all_sc.add(sc)
        return y_pred, y_base, sc_count_pred, sc_count_base, all_sc, key2sc, sc2key

    def print_numdiff(self, sc_count_pred, sc_count_base, all_sc):
        sc_sorted = list(all_sc)
        sc_sorted.sort()

        print("%8s: %8s / %8s" % (" ", "BASE", "PRED"))
        for sc in sc_sorted:
            nsc_pred = 0
            nsc_base = 0
            try:
                nsc_pred = sc_count_pred[sc]
            except:
                pass
            try:
                nsc_base = sc_count_base[sc]
            except:
                pass

            if nsc_base == nsc_pred:
                print("%8d: %8d / %8d     %s" % (sc, nsc_base, nsc_pred, self.scmap[sc]))
            elif nsc_base < nsc_pred:
                if nsc_base == 0:
                    print("%8d: %8d / %8d +++ %s" % (sc, nsc_base, nsc_pred, self.scmap[sc]))
                elif float(nsc_base)/nsc_pred > 0.5:
                    print("%8d: %8d / %8d ++  %s" % (sc, nsc_base, nsc_pred, self.scmap[sc]))
                else:
                    print("%8d: %8d / %8d +   %s" % (sc, nsc_base, nsc_pred, self.scmap[sc]))

            elif nsc_base > nsc_pred:
                if nsc_pred == 0:
                    print("%8d: %8d / %8d --- %s" % (sc, nsc_base, nsc_pred, self.scmap[sc]))
                elif float(nsc_pred)/nsc_base <0.5:
                    print("%8d: %8d / %8d --  %s" % (sc, nsc_base, nsc_pred, self.scmap[sc]))
                else:
                    print("%8d: %8d / %8d -   %s" % (sc, nsc_base, nsc_pred, self.scmap[sc]))
