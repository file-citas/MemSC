#!/usr/bin/env python
import DataPrep
import sys
import os
import random
import math
import pickle
import glob
import pandas as pd
import numpy as np
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

DONE_MAS_FN = "done_mas"

def Writer(dest_filename, some_queue, some_stop_token):
    with open(dest_filename, 'a') as dest_file:
        while True:
            line = some_queue.get()
            if line == some_stop_token:
                return
            dest_file.write(line + '\n')
            dest_file.flush()

def do_bin2csv(param):
    ma_path = param['ma_path']
    annot_path = param['annot_path']
    n = param['n']
    b = param['b']
    ncp = param['ncp']
    do_log = param['do_log']
    outdir = param['outdir']
    f_headers = param['f_headers']
    nosys = param['nosys']
    keepirq = param['keepirq']
    info = param['info']
    done_q = param['done_q']
    drop_irq = not keepirq

    if not os.path.exists(info):
        print("[-] Error: info file %s does not exist" % info)
        return

    info_pref = os.path.splitext(os.path.basename(info))[0] + "_"
    if not os.path.basename(ma_path).startswith(info_pref):
        print('[-] Error key mismatch %s / %s' % (ma_path, info))
    if not os.path.basename(annot_path).startswith(info_pref):
        print('[-] Error key mismatch %s / %s' % (annot_path, info))
    print('[+] Loading\n\t%s\n\t%s\n\t%s' % (ma_path, annot_path, info))
    maf = DataPrep.MaFeatures.MaFeatures()
    maf.add_data(ma_path, info, annot_path=annot_path)
    #features = maf.generate_features(b, n, ncp, do_log)

    typ = "xx"
    if "_ma" in os.path.basename(ma_path):
        typ = "ma"
    elif "_pr" in os.path.basename(ma_path):
        typ = "pr"
    elif "_pw" in os.path.basename(ma_path):
        typ = "pw"
    else:
        print("[-] Error could not determine access type for %s" % ma_path)
        return

    hdf_key = "t_%s_n_%d_b_%d_l_%d" % (typ, n, b, do_log)
    out_path = os.path.join(outdir, "xx_%s" % (os.path.splitext(os.path.basename(ma_path))[0]))
    if os.path.exists(out_path):
        print("[!] Warning removing %s" % out_path)
        os.remove(out_path)
    try:
        if nosys:
            for features in maf.generate_features2_nosys(8192*8, b, n, ncp, do_log):
                if os.path.exists(out_path):
                    features.to_hdf(out_path, key=hdf_key, append=True, format='t',complib='blosc:lz4',complevel=9)
                else:
                    features.to_hdf(out_path, key=hdf_key, format='t', complib='blosc:lz4',complevel=9)
        else:
            for features in maf.generate_features2(b, n, ncp, do_log, drop_irq=drop_irq):
                if os.path.exists(out_path):
                    features.to_hdf(out_path, key=hdf_key, append=True, format='t',complib='blosc:lz4',complevel=9)
                else:
                    features.to_hdf(out_path, key=hdf_key, format='t', complib='blosc:lz4',complevel=9)
    except StopIteration:
        pass

    #features.to_hdf(os.path.join(outdir, "xx_%s" % (os.path.splitext(os.path.basename(ma_path))[0])), key=hdf_key)
    #for k, vdf in features.groupby('KEY'):
    #    #print('[+] Writing feature %x' % k)
    #    try:
    #        #vdf[f_headers].to_csv(os.path.join(outdir, "%s_%x" % (os.path.basename(ma_path), k)), compression='bz2')
    #        #vdf.to_csv(os.path.join(outdir, "%x_%s" % (k, os.path.basename(ma_path))), compression='bz2')
    #        vdf.to_hdf(os.path.join(outdir, "%x_%s" % (k, os.path.splitext(os.path.basename(ma_path))[0])), key=hdf_key)
    #    except Exception as e:
    #        print("[!] Warning invalid dataframe %x_%s" % (k, os.path.basename(ma_path)))
    #        for c in vdf.columns:
    #            print(c)
    #        print(e)
    #        continue
    #    #vdf[f_headers].to_csv(os.path.join(outdir, "%s_%x" % (os.path.basename(ma_path), k)))

    done_q.put(os.path.splitext(os.path.basename(ma_path))[0])



def bin2csv(mas, annots, n, b, ncp, do_log, outdir, infos, nt, nosys, keepirq):

    print('[+] processing memory access blobs')

    m = multiprocessing.Manager()
    queue = m.Queue()
    STOP_TOKEN="STOP!!!"

    writer_process = multiprocessing.Process(target = Writer,
            args=(os.path.join(outdir, DONE_MAS_FN), queue, STOP_TOKEN))
    writer_process.start()

    f_headers = list(map(lambda t: 'MA_%d' % t, range(0, n)))
    f_headers.append('KEY')

    print(annots)
    print(mas)
    with multiprocessing.Pool(nt) as p:
        print(p.map(do_bin2csv, [
            {
                'ma_path': mas[i],
                'annot_path': annots[i],
                'n': n,
                'b': b,
                'ncp': ncp,
                'do_log': do_log,
                'outdir': outdir,
                'f_headers': f_headers,
                'nosys': nosys,
                'keepirq': keepirq,
                'info': infos[i],
                'done_q': queue,
                } for i in range(len(mas))]
            ))
        queue.put(STOP_TOKEN)
        writer_process.join()

def main():
    parser = argparse.ArgumentParser(description='convert extracted csv features to random forrest classifiers')
    parser.add_argument(
        '-i', '--record_id',
        type=str,
        required=True,
        help='ID to be assigned to this record'
    )
    parser.add_argument(
        '-rg', '--record_glob',
        type=str,
        required=True,
        help='glob to include dumps'
    )
    parser.add_argument(
        '-ig', '--info_glob',
        type=str,
        required=True,
        help='glob to include infos'
    )
    parser.add_argument(
        '-c', '-config',
        type=str,
        help='configuration file (Default=./config)',
        default='replay.config'
    )
    parser.add_argument('-ncp',
            help='no class percentage',
            type=float,
            default=1.0
            )
    parser.add_argument('-n',
            type=int,
            default=400,
            help='n features'
            )
    parser.add_argument('-b',
            type=int,
            default=0,
            help='blur'
            )
    parser.add_argument('-nosys',
            action='store_true',
            help='Only generate no sc class'
            )
    parser.add_argument('-keepirq',
            action='store_true',
            help='Keep SCs containing IRQ executions',
            )
    parser.add_argument('-log',
            action='store_true',
            help='Apply log transform to MA offsets'
            )
    parser.add_argument('-nt',
            type=int,
            default=3,
            help='number of threads'
            )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.c)

    ma_out_dir = os.path.join(config['QEMU']['preproc_dir'], '%s_ma_nfeat_%d_blur_%d_log_%d_kirq_%d' % (args.record_id, args.n, args.b, args.log, args.keepirq))
    if not os.path.exists(ma_out_dir):
        os.mkdir(ma_out_dir)
    pr_out_dir = os.path.join(config['QEMU']['preproc_dir'], '%s_pr_nfeat_%d_blur_%d_log_%d_kirq_%d' % (args.record_id, args.n, args.b, args.log, args.keepirq))
    if not os.path.exists(pr_out_dir):
        os.mkdir(pr_out_dir)
    pw_out_dir = os.path.join(config['QEMU']['preproc_dir'], '%s_pw_nfeat_%d_blur_%d_log_%d_kirq_%d' % (args.record_id, args.n, args.b, args.log, args.keepirq))
    if not os.path.exists(pw_out_dir):
        os.mkdir(pw_out_dir)

    ma_done_mas_fn = os.path.join(ma_out_dir, DONE_MAS_FN)
    ma_done_mas = set()
    if os.path.exists(ma_done_mas_fn):
        mal = open(ma_done_mas_fn, "r").readlines()
        for ma in mal:
            ma_done_mas.add(ma.rstrip())

    pr_done_mas_fn = os.path.join(pr_out_dir, DONE_MAS_FN)
    pr_done_mas = set()
    if os.path.exists(pr_done_mas_fn):
        mal = open(pr_done_mas_fn, "r").readlines()
        for ma in mal:
            pr_done_mas.add(ma.rstrip())

    pw_done_mas_fn = os.path.join(pw_out_dir, DONE_MAS_FN)
    pw_done_mas = set()
    if os.path.exists(pw_done_mas_fn):
        mal = open(pw_done_mas_fn, "r").readlines()
        for ma in mal:
            pw_done_mas.add(ma.rstrip())

    #ma_split_dir = os.path.join(config['QEMU']['split_dir'], '%s_ma' % args.record_id)
    #pr_split_dir = os.path.join(config['QEMU']['split_dir'], '%s_pr' % args.record_id)
    #pw_split_dir = os.path.join(config['QEMU']['split_dir'], '%s_pw' % args.record_id)
    ma_split_dir = os.path.join(config['QEMU']['dump_dir']) #, '%s_ma' % args.record_id)
    pr_split_dir = os.path.join(config['QEMU']['dump_dir']) #, '%s_pr' % args.record_id)
    pw_split_dir = os.path.join(config['QEMU']['dump_dir']) #, '%s_pw' % args.record_id)

    print("Reading data from:\n\t%s\n\t%s\n\t%s" % (ma_split_dir, pr_split_dir, pw_split_dir))
    if not os.path.exists(ma_split_dir):
        print("[-] Error: split dir %s does not exist" % ma_split_dir)
        sys.exit(1)
    if not os.path.exists(pr_split_dir):
        print("[-] Error: split dir %s does not exist" % pr_split_dir)
        sys.exit(1)
    if not os.path.exists(pw_split_dir):
        print("[-] Error: split dir %s does not exist" % pw_split_dir)
        sys.exit(1)

    #ma_glob = "%s_ma.*" % (args.record_id)
    #mas_ma = [f for f in \
    #       glob.glob(os.path.join(ma_split_dir, ma_glob)) \
    #       if os.path.isfile(f) and "annotation" not in f and \
    #       os.path.splitext(os.path.basename(f))[0] not in ma_done_mas]
    #ano_ma = [f for f in \
    #       glob.glob(os.path.join(ma_split_dir, ma_glob)) \
    #       if os.path.isfile(f) and "annotation" in f and \
    #       os.path.splitext(os.path.basename(f))[0] not in ma_done_mas]
    #mas_ma.sort()
    #ano_ma.sort()
    #print("[+] Skipping MAs: \n\t%s" % \
    #        '\n\t'.join(ma_done_mas))
    #print("[+] Processing MAs: \n\t%s" % \
    #        '\n\t'.join(map(lambda t: "%16s / %s" % (os.path.basename(t[0]), os.path.basename(t[1])), zip(mas_ma, ano_ma))))
    #bin2csv(mas_ma, ano_ma, args.n, args.b, args.ncp, args.log, ma_out_dir, info, args.nt, args.nosys)

    info_glob = "%s.info" % (args.info_glob)
    infos = [f for f in \
           glob.glob(os.path.join(config['QEMU']['rec_dir'], info_glob)) \
           if os.path.isfile(f) and "info" in f]
    infos.sort()

    mas_pr = []
    ano_pr = []
    infos_pr = []
    for info in infos:
        info_pref = os.path.splitext(os.path.basename(info))[0] + "_"
        pr_glob = "%s*_pr.*" % (info_pref)
        globs = [f for f in \
           glob.glob(os.path.join(pr_split_dir, pr_glob)) \
           if os.path.isfile(f) and "annotation" not in f and \
           os.path.splitext(os.path.basename(f))[0] not in pr_done_mas]
        if len(globs) == 0:
            continue
        if len(globs) != 1:
            print("ERROR too many/few globs for info %s" % info)
            print(globs)
            sys.exit(1)
        ma_pr = globs[0]
        if not os.path.exists(os.path.join(config['QEMU']['dump_dir'], ma_pr)):
            print("Warning %s does not exists" % ma_pr)
            continue
        pr_base = os.path.splitext(os.path.basename(ma_pr))[0]
        ano_name = pr_base + ".annotations"
        if not os.path.exists(os.path.join(config['QEMU']['dump_dir'], ano_name)):
            print("Warning %s does not exists" % ano_name)
            continue
        mas_pr.append(globs[0])
        ano_pr.append(os.path.join(config['QEMU']['dump_dir'], ano_name))
        infos_pr.append(info)

    #pr_glob = "%s_pr.*" % (args.record_glob)
    #mas_pr = [f for f in \
    #       glob.glob(os.path.join(pr_split_dir, pr_glob)) \
    #       if os.path.isfile(f) and "annotation" not in f and \
    #       os.path.splitext(os.path.basename(f))[0] not in pr_done_mas]
    #ano_pr = [f for f in \
    #       glob.glob(os.path.join(pr_split_dir, pr_glob)) \
    #       if os.path.isfile(f) and "annotation" in f and \
    #       os.path.splitext(os.path.basename(f))[0] not in pr_done_mas]
    #mas_pr.sort()
    #ano_pr.sort()
    #infos_pr = []
    #for info in infos:
    #    info_pref = os.path.splitext(os.path.basename(info))[0] + "_"
    #    contained = False
    #    for ma_pr in mas_pr:
    #        if os.path.basename(ma_pr).startswith(info_pref):
    #            contained = True
    #            break
    #    if contained:
    #        infos_pr.append(info)
    print("[+] Processing PRs: \n\t%s" % \
            '\n\t'.join(map(lambda t: "%16s / %s" % (os.path.basename(t[0]), os.path.basename(t[1])), zip(mas_pr, ano_pr))))
    if len(infos_pr) != len(mas_pr) or len(infos_pr) != len(ano_pr):
        print('[-] Error input len mismatch')
        print(infos_pr)
        print(mas_pr)
        print(ano_pr)
        sys.exit(1)
    bin2csv(mas_pr, ano_pr, args.n, args.b, args.ncp, args.log, pr_out_dir, infos_pr, args.nt, args.nosys, args.keepirq)

    mas_pw = []
    ano_pw = []
    infos_pw = []
    for info in infos:
        info_pref = os.path.splitext(os.path.basename(info))[0] + "_"
        pr_glob = "%s*_pw.*" % (info_pref)
        globs = [f for f in \
           glob.glob(os.path.join(pr_split_dir, pr_glob)) \
           if os.path.isfile(f) and "annotation" not in f and \
           os.path.splitext(os.path.basename(f))[0] not in pr_done_mas]
        if len(globs) == 0:
            continue
        if len(globs) != 1:
            print("ERROR too many/few globs for info %s" % info)
            print(globs)
            sys.exit(1)
        ma_pw = globs[0]
        if not os.path.exists(os.path.join(config['QEMU']['dump_dir'], ma_pw)):
            print("Warning %s does not exists" % ma_pw)
            continue
        pr_base = os.path.splitext(os.path.basename(ma_pw))[0]
        ano_name = pr_base + ".annotations"
        if not os.path.exists(os.path.join(config['QEMU']['dump_dir'], ano_name)):
            print("Warning %s does not exists" % ano_name)
            continue
        mas_pw.append(globs[0])
        ano_pw.append(os.path.join(config['QEMU']['dump_dir'], ano_name))
        infos_pw.append(info)


    #pw_glob = "%s_pw.*" % (args.record_glob)
    #mas_pw = [f for f in \
    #       glob.glob(os.path.join(pw_split_dir, pw_glob)) \
    #       if os.path.isfile(f) and "annotation" not in f and \
    #       os.path.splitext(os.path.basename(f))[0] not in pw_done_mas]
    #ano_pw = [f for f in \
    #       glob.glob(os.path.join(pw_split_dir, pw_glob)) \
    #       if os.path.isfile(f) and "annotation" in f and \
    #       os.path.splitext(os.path.basename(f))[0] not in pw_done_mas]
    #mas_pw.sort()
    #ano_pw.sort()
    #infos_pw = []
    #for info in infos:
    #    info_pref = os.path.splitext(os.path.basename(info))[0] + "_"
    #    contained = False
    #    for ma_pw in mas_pw:
    #        if os.path.basename(ma_pw).startswith(info_pref):
    #            contained = True
    #            break
    #    if contained:
    #        infos_pw.append(info)
    print("[+] Processing PWs: \n\t%s" % \
            '\n\t'.join(map(lambda t: "%16s / %s" % (os.path.basename(t[0]), os.path.basename(t[1])), zip(mas_pw, ano_pw))))
    if len(infos_pw) != len(mas_pw) or len(infos_pw) != len(ano_pw):
        print('[-] Error input len mismatch')
        print(infos_pw)
        print(mas_pw)
        print(ano_pw)
        sys.exit(1)
    bin2csv(mas_pw, ano_pw, args.n, args.b, args.ncp, args.log, pw_out_dir, infos_pw, args.nt, args.nosys, args.keepirq)

if __name__ == "__main__":
    #warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    main()
