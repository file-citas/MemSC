#!/usr/bin/env python
import DataPrep
import time
import sys
import os
import random
import math
import pickle
import glob
import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import scipy.stats as stats
from scipy.stats import norm
import multiprocessing
import argparse
import configparser
import warnings
import json
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

NOCLASS = 0xffff
SCMAP_FN = "./scmap.json"
scmap = {}

def load_scmap(scmap_fn=SCMAP_FN):
    tmpmap = json.loads(open(scmap_fn, "r").read())
    tmpmap2 = {}
    for k, v in tmpmap.items():
        tmpmap2[int(k)] = v
    tmpmap2[NOCLASS] = "NOSC"
    return tmpmap2

def get_sc(hdkey):
    parts = hdkey.split("_")
    sc = int(parts[1])
    return sc


def split_hdkey(hdkey):
    parts = hdkey.split("_")
    sckey = "_".join(parts[0:2])
    cokey = "_".join(parts[2:])
    return sckey, cokey

def has_key(out_path, hdkey):
    with h5py.File(out_path, mode='r') as fd:
        if hdkey in fd.keys():
            return True
    return False


def merge_duplicates(in_path, out_path, sc):
    with h5py.File(in_path, mode='r') as fd:
        for hdkey in fd.keys():
            if not hdkey.startswith("sc_%d_" % sc):
                continue
            if hdkey == "INFO":
                continue
            print("Merging %s" % hdkey)
            df = pd.read_hdf(in_path, key=hdkey)
            df = df.reset_index(drop=True)
            c0 = len(df.index)
            cols = [t for t in df.columns.values if t != "Count"]
            df = df.groupby(list(cols)).Count.sum(['Count']).reset_index()
            df = df.drop_duplicates(subset=list(cols), ignore_index=True)
            df.to_hdf(out_path, key=hdkey, format='t', complib='blosc:lz4',complevel=9)
            c1 = len(df.index)
            print("Dedup %d" % (c0-c1))

def merge(out_path, in_paths, nfeat, info):
    processed = []
    npaths = len(in_paths)
    for path_idx, in_path in enumerate(in_paths):
        if "done_mas" in in_path:
            continue
        if in_path in info:
            print("[%6d / %6d] Skip %s" % (path_idx, npaths, in_path))
            continue
        try:
            this_process = {}
            for sc in scmap.keys():
                this_process[sc] = 0
            this_process["path"] = in_path
            with h5py.File(in_path, mode='r') as fd:
                print("[%6d / %6d] Load %s" % (path_idx, npaths, in_path))
                for hdkey in fd.keys():
                    print(hdkey)
                    keys = ["KEY"]
                    keys.extend(["MAD_%d" % i for i in range(1,nfeat)])
                    df = pd.read_hdf(in_path, key=hdkey)[keys]
                    cols = df.columns.values
                    df['Count'] = 1
                    df = df.groupby(list(cols)).Count.count().reset_index()
                    df = df.drop_duplicates(subset=list(cols), ignore_index=True)
                    for sckey in np.unique(df['KEY'].values):
                        hdkeysc = "sc_%d_%s" % (sckey, hdkey)
                        df_sc = df[df['KEY']==sckey]
                        df_sc = df_sc.reset_index(drop=True)
                        this_process[sckey] += len(df_sc.index)
                        if os.path.exists(out_path) and has_key(out_path, hdkeysc):
                            df_sc.to_hdf(out_path, key=hdkeysc, append=True, format='t',complib='blosc:lz4',complevel=9)
                        else:
                            df_sc.to_hdf(out_path, key=hdkeysc, format='t',complib='blosc:lz4',complevel=9)
            processed.append(this_process)
        except Exception as e:
            print("ERROR %s: %s" % (in_path, e))
    return processed

def do_drop(in_fn, out_fn, keep):
    with h5py.File(in_fn, mode='r') as fd:
        for hdkey in fd.keys():
            if hdkey == "INFO":
                continue
            sc = get_sc(hdkey)
            if sc not in keep:
                print("Drop %s" % sc)
                continue
            df = pd.read_hdf(in_fn, key=hdkey)
            assert(len(df[df['KEY'] != sc].index) == 0)
            df.to_hdf(out_fn, key=hdkey, format='t',complib='blosc:lz4',complevel=9)

def expand(in_fn, out_fn, sc):
    with h5py.File(in_fn, mode='r') as fd:
        for hdkey in fd.keys():
            if not hdkey.startswith("sc_%d_" % sc):
                continue
            if hdkey == "INFO":
                continue
            print("Expand %s" % hdkey)
            sckey, cokey = split_hdkey(hdkey)
            df = pd.read_hdf(in_fn, key=hdkey)
            df = df.reset_index(drop=True)
            new_df = df.copy()
            counts = np.unique(df['Count'].values)
            for cnt in counts:
                c_df = df[df['Count'] == cnt]
                for n in range(1, cnt):
                    new_df = new_df.append(c_df, ignore_index=True)
            new_df['Count'] = 1
            new_df = new_df.reset_index(drop=True)
            new_df.to_hdf(out_fn, key=hdkey, format='t', complib='blosc:lz4',complevel=9)
            print("EX %s: %d / %d" % (sckey, len(new_df.index), new_df['Count'].sum()))

def resample(out_fn, in_fn, sc, max_n, min_n):
    with h5py.File(in_fn, mode='r') as fd:
        for hdkey in fd.keys():
            if not hdkey.startswith("sc_%d_" % sc):
                continue
            if hdkey == "INFO":
                continue
            print("Resample %s" % hdkey)
            sckey, cokey = split_hdkey(hdkey)
            df = pd.read_hdf(in_fn, key=hdkey)
            df = df.reset_index(drop=True)
            ndf = df['Count'].sum()
            ndf_u = len(df.index)
            if ndf > max_n and ndf_u > max_n:
                print("Undesample %s %d -> %d" % (sckey, ndf, max_n))
                probs = df['Count'].values
                sum_probs = float(sum(probs))
                probs_norm = [float(i)/sum_probs for i in probs]
                samples = np.random.choice(df.index, max_n, p=probs_norm, replace=False)
                if len(np.unique(samples)) != len(samples):
                    print("ERROR %d / %d / %d / %d samples" % (len(df.index), len(np.unique(df.index)), len(np.unique(samples)), len(samples)))
                try:
                    df = df.iloc[samples]
                except:
                    for samp in samples:
                        try:
                            x = df.iloc[samp]
                        except:
                            print("samp %d oob %d-%d" % (samp, df.index[0], df.index[-1]))
                df['Count'] = 1
                cols = [t for t in df.columns.values if t != "Count"]
                df = df.groupby(list(cols)).Count.count().reset_index()
                df = df.drop_duplicates(subset=list(cols), ignore_index=True)
            elif ndf > max_n and ndf_u > min_n:
                print("Setting count 1")
                df['Count'] = 1
            elif ndf > max_n and ndf_u < min_n:
                print("Setting count 2")
                #df['Count'] = 1
                #mult = int(min_n/ndf_u) + 1
                probs = df['Count'].values
                sum_probs = float(sum(probs))
                probs_norm = [float(i)/sum_probs for i in probs]
                samples = np.random.choice(df.index, max_n, p=probs_norm, replace=True)
                try:
                    df = df.iloc[samples]
                except:
                    for samp in samples:
                        try:
                            x = df.iloc[samp]
                        except:
                            print("samp %d oob %d-%d" % (samp, df.index[0], df.index[-1]))
                df['Count'] = 1
                cols = [t for t in df.columns.values if t != "Count"]
                df = df.groupby(list(cols)).Count.count().reset_index()
                df = df.drop_duplicates(subset=list(cols), ignore_index=True)

            elif ndf > 0 and ndf < min_n:
                mult = int(min_n/ndf) + 1
                print("Oversample %s %d (mult %d) -> %d" % (sckey, ndf, mult, min_n))
                df['Count'] *= mult
                #new_df = df
                #for i in range(0, mult):
                #    new_df = new_df.append(df, ignore_index=True)
                #df = new_df
            df = df.reset_index(drop=True)
            print("RS %s: %d / %d" % (sckey, len(df.index), df['Count'].sum()))
            #if os.path.exists(out_fn) and has_key(out_fn, hdkey):
            #    #df.to_hdf(out_fn, key=hdkey, append=True, format='t')
            #    print("ERROR key %s already exists" % hdkey)
            #else:
            df.to_hdf(out_fn, key=hdkey, format='t', complib='blosc:lz4',complevel=9)

def add_nosys(nosys_fn, out_fn):
    with h5py.File(nosys_fn, mode='r') as fd:
        for hdkey in fd.keys():
            if not hdkey.startswith("sc_%d_" % NOCLASS):
                continue
            if hdkey == "INFO":
                continue
            df = pd.read_hdf(nosys_fn, key=hdkey)
            print("adding %s (%d)" % (hdkey, len(df.index)))
            df.to_hdf(out_fn, key=hdkey, format='t', complib='blosc:lz4',complevel=9)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='merge hdf5 files')
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Path to output hdf5 file'
    )
    parser.add_argument(
        '-drop',
        type=str,
        help="Drop Scs"
    )
    parser.add_argument(
        '-expand',
        action='store_true',
        help="Expand"
    )
    parser.add_argument(
        '-dedup',
        action='store_true',
        help="Deduplicate"
    )
    parser.add_argument(
        '-merge',
        action='store_true',
        help="Add preprocessed feature vectors"
    )
    parser.add_argument(
        '-nosys',
        action='store_true',
        help="Add nosys class"
    )
    parser.add_argument(
        '-sc',
        type=int,
        default=0,
        help="SC class to dedup"
    )
    parser.add_argument(
        '-resample',
        action='store_true',
        help="Resample"
    )
    parser.add_argument(
        '-max',
        type=int,
        default=8192,
        help="Max samples per class"
    )
    parser.add_argument(
        '-nfeat',
        type=int,
        default=512,
        help="Feature Length"
    )
    parser.add_argument(
        '-min',
        type=int,
        default=128,
        help="Min samples per class"
    )
    parser.add_argument(
        '-i', '--input',
        nargs="+",
        help='Path to input hdf5 files'
    )
    args = parser.parse_args()
    scmap = load_scmap()

    if args.nosys:
        add_nosys(args.input[0], args.output)

    if args.drop:
        keep = [int(t) for t in json.loads(open(args.drop, "r").read())]
        do_drop(args.input[0], args.output, set(keep))
        df_info = pd.read_hdf(args.input[0], key="INFO")
        df_info.to_hdf(args.output, key="INFO", format='t',complib='blosc:lz4',complevel=9)
        sys.exit(0)

    if args.expand:
        expand(args.input[0], args.output, args.sc)
        df_info = pd.read_hdf(args.input[0], key="INFO")
        df_info.to_hdf(args.output, key="INFO", format='t',complib='blosc:lz4',complevel=9)
        sys.exit(0)

    if args.dedup:
        merge_duplicates(args.input[0], args.output, args.sc)
        df_info = pd.read_hdf(args.input[0], key="INFO")
        df_info.to_hdf(args.output, key="INFO", format='t',complib='blosc:lz4',complevel=9)
        sys.exit(0)

    if args.resample:
        resample(args.output, args.input[0], args.sc, args.max, args.min)
        df_info = pd.read_hdf(args.input[0], key="INFO")
        df_info.to_hdf(args.output, key="INFO", format='t',complib='blosc:lz4',complevel=9)
        sys.exit(0)

    if args.merge:
        if not os.path.exists(args.output):
            print("Creating %s" % args.output)
            processed = []
            this_process = {}
            for sc in scmap.keys():
                this_process[sc] = 0
            this_process["path"] = "?" * 512
            processed.append(this_process)
            df = pd.DataFrame(data=processed)
            df.to_hdf(args.output, key="INFO", format='t',complib='blosc:lz4',complevel=9)

        info = set()
        df_info = pd.read_hdf(args.output, key="INFO")
        info = set(df_info['path'].values)
        #for info_fn in df_info['path'].values:
        #    info.add(info_fn)

        processed = merge(args.output, args.input, args.nfeat, info)

        df_info = pd.DataFrame(data=processed)
        df_info.to_hdf(args.output, key="INFO", append=True, format='t',complib='blosc:lz4',complevel=9)
