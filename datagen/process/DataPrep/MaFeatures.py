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
import multiprocessing
import json

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import scipy.stats as stats
from scipy.stats import norm

IRQ_PCS_R = [4307552388, 12585092, 4307555621, 12588325, 4294974296, 7000, 4307557035, 12589739]
IRQ_PCS_TEST = [0xc00884, 0x66415, 0x66410, 0xc0088e, 0x25af, 0x25b6]
MAX_SIZE = 1024 * 1024 * 1024 * 16
IDX_DEC = 0
NCOLS = 2
DATA_SIZE = 4

class GetSCs(object):
    def __init__(self, data_ma_path, data_in_path, mdl_path, n=100, typ="pr"):
        self.data_ma_path = data_ma_path
        self.data_in_path = data_in_path
        self.mdl_path = mdl_path
        self.n = n
        self.typ = typ
        self.maf = MaFeatures()
        self.maf.add_data(data_ma_path, data_in_path, annot_path=None, typ=typ)
        self.maei = MaEvalIter(self.maf.df, n)
        self.loaded_model = pickle.load(open(mdl_path, 'rb'))

    def eval(self):
        sc_count = {}
        sc_list = []
        while True:
            idx = self.maei.index
            x = np.empty((0,99), dtype='float64')
            done = False
            for i in range(0, 4096):
                try:
                    x = np.append(x, [self.maei.next()], axis=0)
                except:
                    done = True
                    break
            if done:
                print("DONE")
                break
            y = self.loaded_model.predict(x.astype(np.uint64))
            for i, tmp in enumerate(y):
                if tmp != 65535:
                    sc_name = "???"
                    try:
                        sc_name = eax_sysc_map[tmp]
                    except:
                        pass
                    sc_list.append(sc_name)
                    print("\n%8d: %6x %s" % (idx+i, tmp, sc_name))
                    if tmp not in sc_count.keys():
                        sc_count[tmp] = set()
                    sc_count[tmp].add(idx+1)

        res = {}
        res["sc_list"] = sc_list
        for tmp, idxs in sc_count.items():
            k = "%x" % tmp
            sc_name = "???"
            try:
                sc_name = eax_sysc_map[tmp]
            except:
                pass
            res[k] = {}
            res[k]["sc_name"] = sc_name
            res[k]["count"] = len(idxs)
        return res


class MaEvalIter2(object):
    def __init__(self, maf, step, blur, n, do_log, drop_irq=False, rename_cols=None, filter_keys=None, sc_trans=None):
        self.maf = maf
        self.blur = blur
        self.n = n
        self.do_log = do_log
        self.step = step
        self.start = 0
        self.drop_irq = drop_irq
        self.rename_cols = rename_cols
        self.filter_keys = filter_keys
        self.sc_trans = sc_trans
        self.total = len(self.maf.df.index)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def getIndex(self):
        return self.start

    def getTotal(self):
        return self.total

    def next(self):
        features = self.maf.generate_features_iterative(self.blur, self.n, self.do_log, self.start, self.start+self.step, drop_irq=self.drop_irq)
        if features is None:
            raise StopIteration
        self.start = self.start+self.step
        if self.rename_cols is not None:
            features = features.rename(columns=self.rename_cols)
        if self.filter_keys is not None:
            df_tmp = features
            for fk in self.filter_keys:
                df_tmp = df_tmp[df_tmp['KEY'] != fk]
            features = df_tmp
        if self.sc_trans is not None:
            features = features[features['KEY'].isin(self.sc_trans.keys())]
            for sc0, trans in self.sc_trans.items():
                ftmp = features[features['KEY'] == sc0]
                if len(ftmp.index) > 0:
                    features.loc[features['KEY'] == sc0, 'KEY'] = trans
                    #print(features.iloc[ftmp.index])
        return features

class MaEvalIter(object):
    def __init__(self, df, n_feat):
        self.df = df
        self.n_feat = n_feat
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def getIndex(self):
        return self.index

    def next(self):
        try:
            mas = np.array(self.df.iloc[self.index:self.index+self.n_feat]['MAC'].values, dtype='uint64')
        except:
            raise StopIteration()
        #print(mas)
        mads = np.array([], dtype='uint64')
        for i in range(1, self.n_feat):
            if i>0 and i<len(mas):
                mads = np.append(mads, mas[i] - mas[i-1])
        self.index+=1
        return mads
        #raise StopIteration()

def conf_mat_from_cross_val(X, y, model, class_labels):
    from sklearn.model_selection import KFold
    from sklearn.metrics import confusion_matrix
    """
    Compute the class confusion matrix using cross validation.
    X: the test and train data
    y: the corresponding labels
    model: sklearn classifier
    class_labels: the labels to be shown in the confusion matrix
    """
    result = np.zeros((len(class_labels), len(class_labels)))
    kf = KFold(n_splits=2, shuffle=True)
    for train_index, test_index in kf.split(X):
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       model.fit(X_train, y_train)
       result += confusion_matrix(y_test, model.predict(X_test), labels=class_labels)
    return result

def keys2name(keys, typ,
        shift=0x120, mult=8,
        vmlinux_fn='/data/in/img/vmlinux', base_addr=0xffffffff81000000):
    if typ == 'pr':
        return [id2func(key, shift=shift, mult=mult).lower() for key in keys]
    if typ == 'pw':
        return [addr2func(key, vmlinux_fn=vmlinux_fn, base_addr=base_addr).lower() for key in keys]

def addr2func(addr_offset, vmlinux_fn='/data/in/img/vmlinux', base_addr=0xffffffff81000000):
    if addr_offset == MaFeatures.NOCLASS:
        return 'none'
    p = subprocess.Popen(['/usr/bin/addr2line', '0x%x' % (addr_offset + base_addr), '-e', vmlinux_fn, '-f'],
            stdout=subprocess.PIPE)
    r = str(p.stdout.readline(), 'utf-8')
    return r.rstrip('\n').lower()

def get_model_paths(config, n_ma=None, blur=None, step=None, gap=None, n_tree=None):
    minfos = []
    for f in os.listdir(config['model_dir']):
        if f.endswith('.info'):
            minfo = pickle.load(open(os.path.join(config['model_dir'], f), 'rb'))
            if n_ma is not None and minfo['n_ma'] != n_ma:
                continue
            if blur is not None and minfo['blur'] != blur:
                continue
            if step is not None and minfo['step'] != step:
                continue
            if gap is not None and minfo['gap'] != gap:
                continue
            if n_tree is not None and minfo['n_tree'] != n_tree:
                continue
            minfos.append(minfo)
    return minfos

def get_evals(config, n_ma=None, blur=None, step=None, gap=None, n_tree=None):
    evals = []
    for f in os.listdir(config['eval_dir']):
        if f.endswith('.evl'):
            evl = pickle.load(open(os.path.join(config['eval_dir'], f), 'rb'))
            if n_ma is not None and evl['n_ma'] != n_ma:
                #print('nma %d' % evl['n_ma'])
                continue
            if blur is not None and evl['blur'] != blur:
                #print('blur %d' % evl['blur'])
                continue
            if step is not None and evl['step'] != step:
                #print('step %d' % evl['step'])
                continue
            if gap is not None and evl['gap'] != gap:
                #print('gap %d' % evl['gap'])
                continue
            if n_tree is not None and evl['n_tree'] != n_tree:
                continue
            evals.append(evl)
    return evals


def get_cmat_name(n_ma, blur, step, gap, n_tree):
    cmat_name = 'cmat_n-%d_b-%d_s-%d_g-%d_t-%d' % (n_ma, blur, step, gap, n_tree)
    return cmat_name

def get_model_name(n_ma, blur, step, gap, n_tree):
    model_name = 'model_n-%d_b-%d_s-%d_g-%d_t-%d' % (n_ma, blur, step, gap, n_tree)
    return model_name

def get_eval_name(n_ma, blur, step, gap, n_tree):
    eval_name = 'eval_n-%d_b-%d_s-%d_g-%d_t-%d' % (n_ma, blur, step, gap, n_tree)
    return eval_name

def get_config(config_fn):
    config = configparser.ConfigParser()
    config.read(config_fn)

    try:
        n_trees = list(map(lambda t: int(t), config['BNC']['n_trees'].split(',')))
        sorted(n_trees)
    except:
        n_trees = [10, ]
    print('[+] #Trees: ' + ', '.join(map(lambda t: "%d" % t, n_trees)))


    blurs = list(map(lambda t: int(t), config['BNC']['blurs'].split(',')))
    sorted(blurs)
    blur_max = blurs[-1]
    blur_min = blurs[0]
    print('[+] Blurs: ' + ', '.join(map(lambda t: "%d" % t, blurs)))
    print('[+] Min. Blur %d' % blur_min)
    print('[+] Max. Blur %d' % blur_max)

    n_mas = list(map(lambda t: int(t), config['BNC']['n_mas'].split(',')))
    sorted(n_mas)
    n_ma_max = n_mas[-1]
    n_ma_min = n_mas[0]
    print('[+] #Mem Accesses: ' + ', '.join(map(lambda t: "%d" % t, n_mas)))
    print('[+] Min. #MA %d' % n_ma_min)
    print('[+] Max. #MA %d' % n_ma_max)

    steps = list(map(lambda t: int(t), config['BNC']['steps'].split(',')))
    gaps = list(map(lambda t: int(t), config['BNC']['gaps'].split(',')))
    print('[+] Steps: ' + ', '.join(map(lambda t: "%d" % t, steps)))
    print('[+] Gaps: ' + ', '.join(map(lambda t: "%d" % t, gaps)))

    ncp = config['BNC'].getfloat('ncp')
    print('[+] NCP: %f' % ncp)

    #sc_name_sel = config['BNC']['sc_name_sel'].split(',')
    #print('[+] Selected Syscalls: ' + ', '.join(sc_name_sel))
    sc_key_sel = list(map(lambda t: int(t, 16), config['BNC']['sc_key_sel'].split(',')))
    print('[+] Selected Syscalls: ' + ', '.join(map(lambda t: '%x' % t, sc_key_sel)))

    min_feat = int(config['BNC']['min_feat'])
    max_feat = int(config['BNC']['max_feat'])
    print('[+] Min. #Features: %d' % min_feat)
    print('[+] Max. #Features: %d' % max_feat)

    try:
        balanced = config['BNC'].getboolean('balanced')
    except:
        balanced = False

    if balanced:
        print('[+] Using balanced classes')
    else:
        print('[+] Using unbalanced classes')

    outdir = config['BNC']['outdir']
    testdir = config['BNC']['testdir']
    indirs = config['BNC']['indir'].split(',')


    outdir = config['BNC']['outdir']
    testdir = config['BNC']['testdir']
    testdir_bare = config['BNC']['testdir_bare']
    testdir_info = config['BNC']['testdir_info']
    indirs = config['BNC']['indir'].split(',')

    print('[+] Outdir: %s' % outdir)
    print('[+] Testir: %s' % testdir)
    print('[+] Testdir Bare: %s' % testdir_bare)
    print('[+] Testdir Info: %s' % testdir_info)
    print('[+] Indirs: %s' % ', '.join(indirs))

    #sc_off_fn = config['BNC']['sc_off_fn']
    #sc_names_fn = config['BNC']['sc_names_fn']

    model_dir = os.path.join(outdir, 'models')
    eval_dir = os.path.join(outdir, 'eval')
    return {
            'n_trees': n_trees,
            'blurs': blurs,
            'blur_min': blur_min,
            'blur_max': blur_max,
            'n_mas': n_mas,
            'n_ma_min': n_ma_min,
            'n_ma_max': n_ma_max,
            'steps': steps,
            'gaps': gaps,
            'ncp': ncp,
            #'sc_name_sel': sc_name_sel,
            'sc_key_sel': sc_key_sel,
            'min_feat': min_feat,
            'max_feat': max_feat,
            'outdir': outdir,
            'testdir': testdir,
            'testdir_bare': testdir_bare,
            'testdir_info': testdir_info,
            'indirs': indirs,
            #'sc_off_fn': sc_off_fn,
            #'sc_names_fn': sc_names_fn,
            'model_dir': model_dir,
            'eval_dir': eval_dir,
            }


def Writer(outdir, some_queue, some_stop_token):
    while True:
        k_n_ent = some_queue.get()
        if k_n_ent == some_stop_token:
            return

        key = k_n_ent[0]
        n = k_n_ent[1]
        ents = k_n_ent[2]

        if key == None or ents == None:
            continue

        print("[+] recording entropies for key %x" % key)
        pickle.dump(ents, open(os.path.join(outdir, "ent_n-%d_%x.pkl" % (n, key)), "wb"))


def get_fvec(sc_keys, sc_paths, max_f):
    sc_df = pd.DataFrame()

    df = pd.DataFrame()
    for sc_key in sc_keys:
        sc_df = pd.DataFrame()
        for path in sc_paths[sc_key]:
            #print("[+] loading %s" % path)
            #sc_df = sc_df.append(pd.DataFrame.from_csv(path)[f_headers].transform(lambda t: t >> blur))
            #sc_df = sc_df.append(pd.DataFrame.from_csv(path))
            sc_df = sc_df.append(pd.read_csv(path, compression='bz2'))
            if len(sc_df.index) > max_f:
                print(" [!] enough features for %x" % sc_key)
                break
        df = df.append(sc_df)
    return df


def do_get_entropy(param):
    f_headers = param['f_headers']
    key = param['key']
    paths = param['paths']
    max_f = param['max_f']
    blur = param['blur']
    n_feat = param['n_feat']
    done_q = param['done_q']
    random.shuffle(paths)

    #print("[+] %x %d" % (key, len(paths)))

    sc_df = pd.DataFrame()

    for path in paths:
        #print("[+] loading %s" % path)
        #sc_df = sc_df.append(pd.DataFrame.from_csv(path)[f_headers].transform(lambda t: t >> blur))
        #sc_df = sc_df.append(pd.DataFrame.from_csv(path)[f_headers])
        sc_df = sc_df.append(pd.read_csv(path, compression='bz2')[f_headers])
        if len(sc_df.index) > max_f:
            print(" [!] enough features for %x" % key)
            break

    if blur > 0:
        sc_df['MA_%d' % 0] = sc_df['MA_%d' % 0].transform(lambda t: t >> blur)
    for i in range(1, n_feat):
        if blur > 0:
            sc_df['MA_%d' % i] = sc_df['MA_%d' % i].transform(lambda t: t >> blur)
        sc_df['MAD_%d' % i] = sc_df['MA_%d' % i] - sc_df['MA_%d' % (i - 1)]

    f_headers_mad = ['MAD_%d' % i for i in range(1, n_feat)]
    f2ent = []
    for fh in f_headers_mad:
        _, counts = np.unique(sc_df[fh], return_counts=True)
        probs = counts / sum(counts)
        f2ent.append(stats.entropy(probs))

    done_q.put((key, len(sc_df.index), f2ent))


def get_entropy(
        outdir,
        sc_paths, sc_lens,
        n_feat, blur,
        min_f, max_f,
        nt):

    print('[+] calculating entropy')

    m = multiprocessing.Manager()
    queue = m.Queue()
    STOP_TOKEN="STOP!!!"

    writer_process = multiprocessing.Process(target = Writer,
            args=(outdir, queue, STOP_TOKEN))
    writer_process.start()


    f_headers = ['MA_%d' % i for i in range(0, n_feat)]
    f_headers.append('KEY')

    entropies = None
    with multiprocessing.Pool(nt) as p:
        print(p.map(do_get_entropy, [
            {
                'key': t[0],
                'paths': t[1],
                'f_headers': f_headers,
                'max_f': max_f,
                'blur': blur,
                'n_feat': n_feat,
                'done_q': queue,
                } for t in sc_paths.items() if sc_lens[t[0]] > min_f]
            ))
        queue.put(STOP_TOKEN)
        writer_process.join()




def get_all_sc(indir, sc_off):
    r = []
    for f in os.listdir(indir):
        if f.endswith("%x" % sc_off) or f.endswith("%x.xz" % sc_off):
            r.append(os.path.join(indir, f))
    return r


def get_all_sc_paths(indirs, outdir=None):
    print('[+] analyzing system calls')

    try:
        sc_paths = pickle.load(open(os.path.join(outdir, "sc_paths_all.pkl"), "rb"))
        print('[+] read sc_paths from %s' % os.path.join(outdir, "sc_paths_all.pkl"))
        sc_lens = pickle.load(open(os.path.join(outdir, "sc_lens_all.pkl"), "rb"))
        print('[+] read sc_lens from %s' % os.path.join(outdir, "sc_lens_all.pkl"))
        return (sc_paths, sc_lens)
    except:
        pass

    sc_paths = {}
    sc_lens = {}

    for ind in indirs:
        print('[+] processing %s' % ind)
        for f in os.listdir(ind):
            try:
                f = f.rstrip('.xz')
                sco = int(f.split('_')[-1], 16)
            except:
                continue
            if sco not in sc_paths.keys():
                sc_paths[sco] = []
                sc_lens[sco] = 0
            f_path = os.path.join(ind, f)
            sc_paths[sco].append(f_path)
            num_lines = sum(1 for line in open(f_path))
            sc_lens[sco] += num_lines - 1

    if outdir is not None:
        pickle.dump(sc_paths, open(os.path.join(outdir, "sc_paths_all.pkl"), "wb"))
        pickle.dump(sc_lens, open(os.path.join(outdir, "sc_lens_all.pkl"), "wb"))

    return (sc_paths, sc_lens)


def get_sc_paths(indirs, scos, outdir):
    print('[+] analyzing system calls')

    #try:
    #    sc_paths = pickle.load(open(os.path.join(outdir, "sc_paths.pkl"), "rb"))
    #    print('[+] read sc_paths from %s' % os.path.join(outdir, "sc_paths.pkl"))
    #    sc_lens = pickle.load(open(os.path.join(outdir, "sc_lens.pkl"), "rb"))
    #    print('[+] read sc_lens from %s' % os.path.join(outdir, "sc_lens.pkl"))
    #    return (sc_paths, sc_lens)
    #except:
    #    pass
    sc_paths = {}
    sc_lens = {}
    #for sco in sc_off_sel[33:-3]:
    for sco in scos:
        sc_paths[sco] = []
        sc_lens[sco] = 0
        for ind in indirs:
            sc_paths[sco].extend(get_all_sc(ind, sco))
            # XXXX
            sc_lens[sco] = 6000000
            #for d in get_all_sc(ind, sco):
            #    num_lines = sum(1 for line in open(d))
            #    #l = len(open(d, "r").readlines())
            #    sc_lens[sco] += num_lines - 1


    sco = MaFeatures.NOCLASS
    sc_paths[sco] = []
    sc_lens[sco] = 0
    for ind in indirs:
        sc_paths[sco].extend(get_all_sc(ind, sco))
        # XXXX
        sc_lens[sco] = 6000000
        #for d in get_all_sc(ind, sco):
        #    l = len(open(d, "r").readlines())
        #    sc_lens[sco] += l

    pickle.dump(sc_paths, open(os.path.join(outdir, "sc_paths.pkl"), "wb"))
    pickle.dump(sc_lens, open(os.path.join(outdir, "sc_lens.pkl"), "wb"))

    return (sc_paths, sc_lens)


def get_scs(fn_scn, fn_sco):
    sc_offs = list(map(lambda t: int(t, 16), open(fn_sco, "r").readlines()))
    sc_names = list(map(lambda t: t[1:-3], open(fn_scn, "r").readlines()))
    sc_o2n = {500: 'none'}
    sc_n2o = {'none': 500}
    for o, n in zip(sc_offs, sc_names):
        sc_o2n[o] = n.lower()
        sc_n2o[n.lower()] = o

    return (sc_offs, sc_names, sc_o2n, sc_n2o)


#import matplotlib.pyplot as plt
#def plot_confusion_matrix(cm, classes,
#                          normalize=False,
#                          title='Confusion matrix',
#                          cmap=plt.cm.Blues):
#    """
#    This function prints and plots the confusion matrix.
#    Normalization can be applied by setting `normalize=True`.
#    """
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#
#    print(cm)
#
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)
#
#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black", fontsize=4)
#
#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
#    #ax.tick_params(axis='both', which='major', pad=15)

def compress(path, typ, verbose=False):
    if verbose:
        print('[+] compresing %s with %s' % (path, typ))
    if typ == '7z':
        subprocess.run(['p7zip', path])
    elif typ == 'xz':
        subprocess.run(['xz', '-z', path])
    else:
        print("[!] Error invalid format use .xz or .7z")


def decompress(path):
    new_file, filename = tempfile.mkstemp(prefix="madecomp_")
    if path.endswith('.7z'):
        print('[+] decompressing 7z %s to %s' % (path, filename))
        p7zcmd = ['p7zip', '-d', '-c', path]
        print('[ ] run %s' % ' '.join(p7zcmd))
        subprocess.run(p7zcmd, stdout=new_file)
    elif path.endswith('.xz'):
        print('[+] decompressing xz %s to %s' % (path, filename))
        subprocess.run(['xz', '-d', '-c', path], stdout=new_file)
    else:
        print('[+] do not decompress %s' % (path))
        return (None, path)
    return (new_file, filename)

class MaFeatures(object):
    NOCLASS = 0xffff
    COLUMNS_DATA =  ['PC', 'MAC']
    COLUMNS_ANOT =  ['KEY', 'IDX_0']
    def __init__(self):
        self.df = pd.DataFrame(columns = MaFeatures.COLUMNS_DATA)
        self.df['MAC'] = self.df['MAC'].astype('uint32')
        self.df['PC'] = self.df['PC'].astype('uint32')
        self.annot_df = pd.DataFrame(columns = MaFeatures.COLUMNS_ANOT)
        self.annot_df['KEY'] = self.annot_df['KEY'].astype('uint16')
        self.annot_df['IDX_0'] = self.annot_df['IDX_0'].astype('uint64')
        self.tmp_fds = {}

    #def __del__(self):
    #    for tfs_name, tfs_fd in self.tmp_fds.items():
    #        if tfs_fd is not None:
    #            try:
    #                os.close(tfs_fd)
    #                if os.path.exists(tfs_name):
    #                    os.remove(tfs_name)
    #            except:
    #                pass

    def __filepos_to_arrayindex(self, filepos):
        return (int) (filepos / (NCOLS* DATA_SIZE))

    def add_ma(self, ma_path, info_path):
        print('[+] Loading Memory Access File %s' % ma_path)
        ma_fd, ma_path_new = decompress(ma_path)
        if ma_path != ma_path_new:
            ma_path = ma_path_new
            print('[+] Loading Decompressed Memory Access File %s' % ma_path)
            self.tmp_fds[ma_path] = ma_fd
        dump_size = os.stat(ma_path).st_size # in bytes
        if dump_size > MAX_SIZE:
            print('[!] Error file too big %d (max %d)' % (dump_size, MAX_SIZE))
            raise Exception("File too big")
        n_cols = NCOLS
        n_rows = int(dump_size / (n_cols * DATA_SIZE))

        new_data = np.memmap(
            ma_path,
            dtype=np.uint32,
            mode='c',
            shape=(n_rows, n_cols))

        start_idx = len(self.df.index)
        stop_idx = start_idx + n_rows
        print('[+] new index %d-%d' % (start_idx, stop_idx))
        new_df = pd.DataFrame(
                data=new_data,
                columns=MaFeatures.COLUMNS_DATA,
                index=range(start_idx, stop_idx),
                dtype=np.uint32)


        info = pickle.load(open(info_path, "rb"))
        #print(json.dumps(info, indent=2))
        print("[+] sc table pa %x" % info['sctablepa'])
        #kstext = info['kstext'] - 0xffffffff80000000
        kstext = info['kstext'] - 0xffffffff81000000
        print("[+] text offset %x" % kstext)
        #new_df['KEY'] = new_df['KEY'] - kstext

        #new_df['KEY'] = new_df['KEY'].astype('uint64')
        new_df['MAC'] = new_df['MAC'].astype('uint32')
        kstext = np.repeat(info['kstext'], stop_idx - start_idx).astype('uint64')
        new_df['PC'] = new_df['PC'] - kstext
        new_df['PC'] = new_df['PC'].astype('uint32')

        if ma_fd is not None:
            os.close(ma_fd)
        if ma_path.startswith("/tmp/madecomp_") and os.path.exists(ma_path):
            os.remove(ma_path)
        return new_df

    def add_annot(self, annot_path):
        print("[+] Loading Annotation File: %s" % annot_path)
        annots = json.load(open(annot_path, 'r'))

        new_annot_data = []
        for a in annots:
            filepos = a["ma"]
            idx = self.__filepos_to_arrayindex(filepos)
            if idx >= IDX_DEC:
                idx -= IDX_DEC
            else:
                idx = 0
            if idx > self.df.index[-1]:
                print("ERROR: pos %d -> idx %d > %d" % (filepos, idx, self.df.index[-1]))
                return
            new_annot_data.append({
                'KEY': a['callno'],
                'IDX_0': idx
                })

        n_rows = len(new_annot_data)

        try:
            start_idx = self.annot_df.index[-1]
        except:
            start_idx = 0
        stop_idx = start_idx + n_rows
        new_annot_df = pd.DataFrame(
                data=new_annot_data,
                columns=MaFeatures.COLUMNS_ANOT,
                index=range(start_idx, stop_idx))

        new_annot_df['KEY'] = new_annot_df['KEY'].astype('uint16')
        new_annot_df['IDX_0'] = new_annot_df['IDX_0'].astype('uint64')
        if len(new_annot_df.index) > 0:
            print('[+] Got %d annotated features' % new_annot_df.index[-1])
        else:
            print('[-] Warning: Got %d annotated features' % 0)
        return new_annot_df


    def add_data(self, ma_path, info_path, annot_path=None):
        new_df = self.add_ma(ma_path, info_path)
        self.df = self.df.append(new_df)
        if annot_path is not None:
            new_annot_df = self.add_annot(annot_path)
            self.annot_df = self.annot_df.append(new_annot_df)

    def get_block_features(self):
        new_df = self.df.copy()
        new_df['sentinel'] = (new_df['KEY'] != new_df['KEY'].shift(1)).astype(int).cumsum()
        new_df = new_df.groupby(['sentinel', 'KEY'])['MAC'].apply(list)
        return new_df

    def generate_features_iterative(self, blur, n, do_log, start, stop, drop_irq=False):
        if start >= self.df.index[-1]:
            return None
        new_df = self.annot_df[(self.annot_df['IDX_0'] >= start) & (self.annot_df['IDX_0'] < stop)].copy()
        new_df['IDX_0'] = new_df['IDX_0'].astype('uint64')
        new_df['KEY'] = new_df['KEY'].astype('uint16')
        new_df2 = pd.DataFrame({"IDX_0": tuple(range(start, stop))})
        new_df3 = pd.merge(new_df2, new_df, on=["IDX_0"], how='left')
        new_df3['KEY'] = new_df3['KEY'].fillna(MaFeatures.NOCLASS)
        #new_df2 = new_df.set_index('IDX_0').reindex(
        #        range(start, stop), fill_value=MaFeatures.NOCLASS
        #        ).reset_index(drop=True)
        new_df3['IDX_0'] = new_df3['IDX_0'].astype('uint64')
        new_df3['KEY'] = new_df3['KEY'].astype('uint16')
        #print(new_df3)
        #print(new_df3[new_df3['KEY'] != MaFeatures.NOCLASS])
        return self.__compute_features(new_df3, blur, n, do_log, drop_irq=drop_irq)

    def generate_features2_nosys(self, n_nosys, blur, n, noclass_perc, do_log):
        all_idx = set(self.df.index)
        #n2 = int(n/2) + 16
        n2 = 1
        #if n2 > n:
        #    n2 = int(n/2)
        for feature_idx in self.annot_df['IDX_0'].values:
            start = 0
            if feature_idx > n2:
                start = int(feature_idx - n2)
            for idx in range(start, start+n2):
                try:
                    all_idx.remove(idx)
                except:
                    pass

        if n_nosys > len(all_idx):
            n_nosys = len(all_idx) - 1
        print("[+] remaining indices %d/%d" % (len(all_idx), len(set(self.df.index))))
        new_df = pd.DataFrame([
            {'KEY': MaFeatures.NOCLASS, 'IDX_0': i} for i in random.sample(tuple(all_idx), n_nosys)],
            index=range(0, n_nosys))
        new_df['IDX_0'] = new_df['IDX_0'].astype('uint64')
        new_df = new_df.reset_index(drop=True)

        step = 16384 * 2
        start = 0
        stop = step
        feature_df = None
        while True:
            if start >= len(new_df.index):
                break
            if stop > len(new_df.index):
                stop = len(new_df.index)
            print('[+] Computing features %d:%d' % (start, stop))
            tmpbase_df = new_df.iloc[start:stop].reset_index(drop=True)
            tmpdf = self.__compute_features(tmpbase_df, blur, n, do_log, drop_irq=False)
            yield tmpdf
            start += step
            stop  += step
        #raise StopIteration


    def generate_features2(self, blur, n, noclass_perc, do_log, drop_irq=True):
        new_df = self.annot_df.copy()
        new_df['IDX_0'] = new_df['IDX_0'].astype('uint64')
        new_df = new_df.reset_index(drop=True)

        step = 16384 * 2
        start = 0
        stop = step
        feature_df = None
        while True:
            if start >= len(new_df.index):
                break
            if stop > len(new_df.index):
                stop = len(new_df.index)
            print('[+] Computing features %d:%d' % (start, stop))
            tmpbase_df = new_df.iloc[start:stop].reset_index(drop=True)
            #for x in tmpbase_df.columns:
            #    tmpbase_df[x] = tmpbase_df[x].astype('uint64')
            tmpdf = self.__compute_features(tmpbase_df, blur, n, do_log, drop_irq=drop_irq)
            yield tmpdf
            #if feature_df is not None:
            #    feature_df = feature_df.append(tmpdf, ignore_index=True)
            #else:
            #    feature_df = tmpdf
            start += step
            stop  += step
        #return self.__compute_features(new_df, blur, n, do_log)
        #return feature_df.reset_index(drop=True)


    def generate_features(self, blur, n, noclass_perc, do_log, drop_irq=False, dbg=False):
        new_df = self.annot_df.copy()
        #try:
        #    last_idx = new_df.index[-1]+1
        #    n_noclass = int(new_df.index[-1] * noclass_perc)
        #    if n_noclass == 0:
        #        n_noclass = 2
        #    print('[+] Adding %d / %d noclass features' % (n_noclass, new_df.index[-1]))
        #except IndexError:
        #    last_idx = 0
        #    n_noclass = 100
        #    print('[+] Adding %d / %d noclass features' % (n_noclass, 0))

        #new_df = pd.concat((new_df,
        #    pd.DataFrame([
        #        {'KEY': MaFeatures.NOCLASS, 'IDX_0': i} for i in np.random.randint(self.df.index[-1]-n, size=(n_noclass))],
        #        index=range(last_idx, last_idx+n_noclass))), axis=0)
        new_df['IDX_0'] = new_df['IDX_0'].astype('uint64')
        new_df = new_df.reset_index(drop=True)

        step = 16384 * 2
        start = 0
        stop = step
        feature_df = None
        while True:
            if start >= len(new_df.index):
                break
            if stop > len(new_df.index):
                stop = len(new_df.index)
            print('[+] Computing features %d:%d' % (start, stop))
            tmpbase_df = new_df.iloc[start:stop].reset_index(drop=True)
            #for x in tmpbase_df.columns:
            #    tmpbase_df[x] = tmpbase_df[x].astype('uint64')
            tmpdf = self.__compute_features(tmpbase_df, blur, n, do_log, drop_irq=drop_irq, dbg=dbg)
            if feature_df is not None:
                feature_df = feature_df.append(tmpdf, ignore_index=True)
            else:
                feature_df = tmpdf
            start += step
            stop  += step
        #return self.__compute_features(new_df, blur, n, do_log)
        return feature_df.reset_index(drop=True)

    def __compute_features(self, new_df, blur, n, do_log, drop_irq=True, dbg=False):
        idx_data = {}
        for i in range(1, n):
            #new_df['IDX_%d' % i] = new_df['IDX_0'] + i
            idx_data['IDX_%d' % i] = (new_df['IDX_0'] + i).values
        idx_df = pd.DataFrame(data=idx_data)
        new_df = pd.concat((new_df, idx_df), axis=1)

        last_idx = "IDX_%d" % (n-1)
        if new_df[last_idx].max() > self.df.index[-1]:
            print('[-] Warning Last idx %d > %d' % (new_df[last_idx].max(), self.df.index[-1]))
            new_df = new_df[new_df[last_idx] <= self.df.index[-1]]

        #print("[+] Last idx: %d / %d" % (new_df[last_idx].max(), self.df.index[-1]))
        new_df = new_df.reset_index(drop=True)

        ma_data = {}
        for i in range(0, n):
            #new_df['MA_%d' % i] = self.df.iloc[new_df['IDX_%d' % i]]['MAC'].values
            try:
                ma_data['MA_%d' % i] = self.df.iloc[new_df['IDX_%d' % i]]['MAC'].values
                ma_data['PC_%d' % i] = self.df.iloc[new_df['IDX_%d' % i]]['PC'].values
            except Exception as e:
                print("XXXXXXXXXXXXXXXXXX")
                print(new_df)
                print(i)
                print(e)
        ma_df = pd.DataFrame(data=ma_data)
        if drop_irq:
            #irq_idx = list(ma_df[ma_df.isin([4307552422]).any(axis=1)].index)
            #irq_idx.extend(list(ma_df[ma_df.isin([4307552431]).any(axis=1)].index))
            #pcrange = ['PC_%d' % t for t in range(0,128)]
            #irq_idx = list(ma_df[ma_df[pcrange].isin([IRQ_PCS_TEST[0]]).any(axis=1)].index)
            #irq_idx.extend(list(ma_df[ma_df[pcrange].isin([IRQ_PCS_TEST[1]]).any(axis=1)].index))
            #irq_idx.extend(list(ma_df[ma_df[pcrange].isin([IRQ_PCS_TEST[2]]).any(axis=1)].index))
            #irq_idx.extend(list(ma_df[ma_df[pcrange].isin([IRQ_PCS_TEST[3]]).any(axis=1)].index))
            #irq_idx.extend(list(ma_df[ma_df[pcrange].isin([IRQ_PCS_TEST[4]]).any(axis=1)].index))
            #irq_idx.extend(list(ma_df[ma_df[pcrange].isin([IRQ_PCS_TEST[5]]).any(axis=1)].index))
            irq_idx = list(ma_df[ma_df.isin([4307552388]).any(axis=1)].index)
            irq_idx.extend(list(ma_df[ma_df.isin([12585092]).any(axis=1)].index))
            irq_idx.extend(list(ma_df[ma_df.isin([4307555621]).any(axis=1)].index))
            irq_idx.extend(list(ma_df[ma_df.isin([12588325]).any(axis=1)].index))
            irq_idx.extend(list(ma_df[ma_df.isin([4294974296]).any(axis=1)].index))
            irq_idx.extend(list(ma_df[ma_df.isin([7000]).any(axis=1)].index))
            irq_idx.extend(list(ma_df[ma_df.isin([4307557035]).any(axis=1)].index))
            irq_idx.extend(list(ma_df[ma_df.isin([12589739]).any(axis=1)].index))
            if len(irq_idx) > 0:
                nosc_keys = new_df[new_df['KEY'] == MaFeatures.NOCLASS].reset_index(drop=True)
                irq_idx_nosc = []
                for iidx in irq_idx:
                    if new_df.iloc[iidx]['KEY'] != MaFeatures.NOCLASS:
                        irq_idx_nosc.append(iidx)
                print("[+] Dropping %d(%d) / %d rows due to irq" % (len(irq_idx_nosc), len(irq_idx), len(ma_df.index)))
                if dbg:
                    self.dropped = ma_df.iloc[irq_idx_nosc]
                ma_df.drop(ma_df.index[irq_idx_nosc], inplace=True)
                ma_df = ma_df.reset_index(drop=True)
                new_df.drop(new_df.index[irq_idx_nosc], inplace=True)
                new_df = new_df.reset_index(drop=True)

        #pcs = set(np.unique(self.df.iloc[new_df['IDX_%d' % i]]['PC'].values))
        #if 4307555621 in pcs or 4307552388 in pcs:
        #    continue
        new_df = pd.concat((new_df, ma_df), axis=1)

        if blur > 0:
            #print("[+] Apply blur %d to MAs" % blur)
            cols = ['MA_%d' % i for  i in range(0, n)]
            new_df[cols] = new_df[cols].applymap(lambda t: t >> blur)
            #for i in range(0, n):
            #    new_df['MA_%d' % i] = new_df['MA_%d' % i].transform(lambda t: t >> blur)

        for x in new_df.columns:
            new_df[x] = new_df[x].astype('uint64')

        mad_data = {}
        for i in range(1, n):
            mad_data['MAD_%d' % i] = (new_df['MA_%d' % i].astype('int64') - new_df['MA_%d' % (i - 1)].astype('int64')).values
            #new_df['MAD_%d' % i] = new_df['MA_%d' % i].astype('int64') - new_df['MA_%d' % (i - 1)].astype('int64')
        mad_df = pd.DataFrame(data=mad_data)
        new_df = pd.concat((new_df, mad_df), axis=1)

        if do_log:
            #print("[+] Apply log2 to offsets")
            cols = ['MAD_%d' % i for  i in range(1, n)]
            df_0 = np.log2(new_df[cols].replace(0, np.nan))
            df_1 = -1 * np.log2(-new_df[cols].replace(0, np.nan))
            df_2 = df_0.fillna(df_1)
            new_df[cols] = df_2.fillna(0)

        new_df['KEY'] = new_df['KEY'].astype('uint16')
        new_cols = ['KEY']
        new_cols.extend(['MAD_%d' % i for  i in range(1, n)])
        new_cols.extend(['MA_%d' % i for  i in range(0, n)])
        new_cols.extend(['PC_%d' % i for  i in range(0, n)])
        return new_df[new_cols]
        #return new_df


#    def get_annot_features_step_gap(self, gap, step, noclass_perc=2.5, n=20):
#        if step <= gap:
#            print("Ivalid step/gap value: step=%d, gap=%d\n", step, gap)
#            return None
#
#        new_df = self.annot_df.copy()
#        n_noclass = int(new_df.index[-1] * noclass_perc)
#        new_df = new_df.append(
#            pd.DataFrame([
#                {'KEY': MaFeatures.NOCLASS, 'IDX_0': i} for i in np.random.randint(self.df.index[-1], size=(n_noclass))],
#                index=range(new_df.index[-1] +1, new_df.index[-1] +1 + n_noclass)))
#
#        idx = 0
#        for i in range(0, n):
#            if (idx % step) == 0:
#                idx += gap
#            new_df['IDX_%d' % i] = new_df['IDX_0'] + idx
#            idx += 1
#
#        for i in range(0, n):
#            new_df['MA_%d' % i] = self.df.iloc[new_df['IDX_%d' % i]]['MAC'].values
#
#        for i in range(1, n):
#            new_df['MAD_%d' % i] = new_df['MA_%d' % i] - new_df['MA_%d' % (i - 1)]
#
#        return new_df

    def get_annot_features4(self, sc_offsets, start, n, n_items=1000, n_step=2):
        r_df = pd.DataFrame(columns=['KEY', 'IDX_0'], dtype=np.uint64)
        stop = start + n_items * n_step
        if not hasattr(self, "sc_indices"):
            self.sc_indices = self.df[self.df['KEY'].isin(sc_offsets)].index

        sci_marker = 0
        while self.sc_indices[sci_marker] < start:
            sci_marker += 1

        df_idx = self.df.iloc[start:stop:n_step].index
        r_df = pd.DataFrame(columns=['KEY', 'IDX_0'], data={'KEY': MaFeatures.NOCLASS, 'IDX_0': df_idx}, index=df_idx)
        for dfi in df_idx:
            if dfi >= self.sc_indices[sci_marker]:
                sci_marker += 1
                r_df.loc[dfi]['KEY'] = self.df.loc[self.sc_indices[sci_marker]]['KEY']

        r_df['KEY'] = r_df['KEY'].astype('uint64')
        r_df['IDX_0'] = r_df['IDX_0'].astype('uint64')

        for i in range(1, n):
            r_df['IDX_%d' % i] = r_df['IDX_0'] + i
            r_df['IDX_%d' % i] = r_df['IDX_%d' % i].astype('uint64')

        for i in range(0, n):
            r_df['MA_%d' % i] = self.df.iloc[r_df['IDX_%d' % i]]['MAC'].astype('uint64').values

        for i in range(1, n):
            r_df['MAD_%d' % i] = r_df['MA_%d' % i].astype('int64') - r_df['MA_%d' % (i - 1)].astype('int64')

        return (r_df, stop)


    def get_index_keys_write(self, n, sc_offset=0xe036c0):
        start_indices =  [i + 0 for i in np.flatnonzero(self.df['KEY'] == sc_offset) if i < len(self.df) - n]
        start_indices1 = [i + 1 for i in start_indices]
        keys = self.df.iloc[start_indices1]['KEY'].values
        return (start_indices, keys)

    def get_index_keys_read(self, n, sc_offset=0xe036c0, apic_int=0x949c07):
        start_indices =  [i + 0 for i in np.flatnonzero(self.df['KEY'] == sc_offset) if i < len(self.df) - n]
        start_indices = [i for i in start_indices if self.df.iloc[i+1]['KEY'] != apic_int]
        keys = self.df.iloc[start_indices]['MAC'].transform(lambda t: t & 0xfff).values
        return (start_indices, keys)

    #def get_feature_vectors(self, typ, n, noclass_perc, sc_offset=0x1b72d20):
    def get_feature_vectors(self, typ, n, noclass_perc, sc_offset=0xe036c0):
        start_indices = []
        keys = []
        print('[+] Extracting feature start points %s' % typ)
        if typ == 'pw':
            start_indices, keys = self.get_index_keys_write(n, sc_offset=sc_offset)
        elif typ == 'pr':
            start_indices, keys = self.get_index_keys_read(n, sc_offset=sc_offset)
        else:
            print('[-] Error unkown type %s' % typ)
            return None

        r_df = pd.DataFrame(columns=['KEY', 'IDX_0'], data={'KEY': keys, 'IDX_0': start_indices})
        if len(r_df.index) == 0:
            print(r_df)
            return None

        sc_indices = set(range(len(self.df.index) - n - 1))
        for idx in start_indices:
            sc_indices -= set(range(max(0, idx - n), idx + n))

        n_noclass = int(len(r_df.index) * noclass_perc)
        print('[+] adding %d noclass features' % n_noclass)
        new_df = \
            pd.DataFrame([
                {'KEY': MaFeatures.NOCLASS, 'IDX_0': i} for i in random.sample(sc_indices, n_noclass)],
                index=range(len(r_df.index), len(r_df.index) + n_noclass))
        r_df = r_df.append(new_df, sort=False)

        print('[+] adjusting data frame types')
        r_df['KEY'] = r_df['KEY'].astype('uint64')
        r_df['IDX_0'] = r_df['IDX_0'].astype('uint64')

        print('[+] extending indices')
        for i in range(1, n):
            r_df['IDX_%d' % i] = r_df['IDX_0'] + i
            r_df['IDX_%d' % i] = r_df['IDX_%d' % i].astype('uint64')

        print('[+] extracting memory accesses')
        for i in range(0, n):
            r_df['MA_%d' % i] = self.df.iloc[r_df['IDX_%d' % i]]['MAC'].astype('uint64').values
            r_df['IP_%d' % i] = self.df.iloc[r_df['IDX_%d' % i]]['KEY'].astype('uint64').values

        return r_df


    def get_annot_features3(self, sc_offsets, noclass_perc=2.5, n=20, blur=0, do_diff=False):
        r_df = pd.DataFrame(columns=['KEY', 'IDX_0'], dtype=np.uint64)
        starti = 0
        stopi = 0
        sc_indices = set(range(len(self.df.index) - n - 1))
        sc_cntr = 0
        for sc_off in sc_offsets:

            #print('[+] adding offset %x (%d/%d)' % (sc_off, sc_cntr, len(sc_offsets)))
            sc_cntr += 1

            data = np.flatnonzero(self.df['KEY'] == sc_off)
            for idx in data:
                sc_indices -= set(range(max(0, idx - n), idx + n))
            stopi = starti + len(data)
            new_df = pd.DataFrame(columns=['KEY', 'IDX_0'], data={'KEY': sc_off, 'IDX_0': data}, index=range(starti, stopi))
            drop_indices = new_df[new_df['IDX_0'] >= len(self.df.index) - n - 1].index
            if not drop_indices.empty:
                print('[!] dropping %d of %d indices for sc %x' % (len(drop_indices), len(new_df.index), sc_off))
                new_df = new_df.drop(drop_indices)
            new_df['KEY'] = new_df['KEY'].astype('uint64')
            new_df['IDX_0'] = new_df['IDX_0'].astype('uint64')
            r_df = r_df.append(new_df)
            starti = stopi

        if len(r_df.index) == 0:
            return None

        r_df['KEY'] = r_df['KEY'].astype('uint64')
        r_df['IDX_0'] = r_df['IDX_0'].astype('uint64')

        n_noclass = int(len(r_df.index) * noclass_perc)
        print('[+] adding %d noclass features' % n_noclass)
        new_df = \
            pd.DataFrame([
                {'KEY': MaFeatures.NOCLASS, 'IDX_0': i} for i in random.sample(sc_indices, n_noclass)],
                index=range(len(r_df.index), len(r_df.index) + n_noclass))
        r_df = r_df.append(new_df, sort=False)

        r_df['KEY'] = r_df['KEY'].astype('uint64')
        r_df['IDX_0'] = r_df['IDX_0'].astype('uint64')

        for i in range(1, n):
            r_df['IDX_%d' % i] = r_df['IDX_0'] + i
            r_df['IDX_%d' % i] = r_df['IDX_%d' % i].astype('uint64')

        for i in range(0, n):
            r_df['MA_%d' % i] = self.df.iloc[r_df['IDX_%d' % i]]['MAC'].astype('uint64').values
            if blur > 0:
                r_df['MA_%d' % i] = r_df['MA_%d' % i].transform(lambda t: t >> blur)
            r_df['IP_%d' % i] = self.df.iloc[r_df['IDX_%d' % i]]['KEY'].astype('uint64').values

        if do_diff:
            for i in range(1, n):
                r_df['MAD_%d' % i] = r_df['MA_%d' % i].astype('int64') - r_df['MA_%d' % (i - 1)].astype('int64')

        return r_df


#    def get_annot_features2(self, noclass_perc=2.5, n=20):
#        new_df = self.annot_df.copy()
#        n_noclass = int(new_df.index[-1] * noclass_perc)
#        new_df = new_df.append(
#            pd.DataFrame([
#                {'KEY': MaFeatures.NOCLASS, 'IDX_0': i} for i in np.random.randint(self.df.index[-1], size=(n_noclass))],
#                index=range(new_df.index[-1] +1, new_df.index[-1] +1 + n_noclass)), sort=False)
#        for i in range(1, n):
#            new_df['IDX_%d' % i] = new_df['IDX_0'] + i
#
#        for i in range(0, n):
#            try:
#                new_df['MA_%d' % i] = self.df.iloc[new_df['IDX_%d' % i]]['MAC'].values
#            except:
#                continue
#
#        for i in range(1, n):
#            try:
#                new_df['MAD_%d' % i] = new_df['MA_%d' % i] - new_df['MA_%d' % (i - 1)]
#            except:
#                continue
#
#        for x in new_df.columns:
#            new_df[x] = new_df[x].astype('uint64')
#        return new_df

    def get_annot_features(self, noclass_perc=2.5, n=20):
        fdata = []

        for index, row in self.annot_df.iterrows():
            f = [row['KEY']]
            for i in range(1, n):
                f.append(self.df.iloc[row['IDX_0'] + i]['MAC'] - self.df.iloc[row['IDX_0'] + i - 1]['MAC'])
            fdata.append(f)

        n_noclass = int(self.annot_df.index[-1] * noclass_perc)
        for i in range(n_noclass):
            f = [MaFeatures.NOCLASS]
            ridx = random.randint(0, self.data_start_idx - n)
            for j in range(1, n):
                f.append(self.df.iloc[ridx + j]['MAC'] - self.df.iloc[ridx + j - 1]['MAC'])
            fdata.append(f)

        c = ['KEY']
        c += ["MA%d" % i for i in range(n)]
        new_df = pd.DataFrame(data=fdata, columns=c)
        for x in new_df.columns:
            new_df[x] = new_df[x].astype('uint64')
        return new_df


eax_sysc_map = {}
eax_sysc_map[0] = "sys_read"
eax_sysc_map[1] = "sys_write"
eax_sysc_map[2] = "sys_open"
eax_sysc_map[3] = "sys_close"
eax_sysc_map[4] = "sys_newstat"
eax_sysc_map[5] = "sys_newfstat"
eax_sysc_map[6] = "sys_newlstat"
eax_sysc_map[7] = "sys_poll"
eax_sysc_map[8] = "sys_lseek"
eax_sysc_map[9] = "sys_mmap"
eax_sysc_map[10] = "sys_mprotect"
eax_sysc_map[11] = "sys_munmap"
eax_sysc_map[12] = "sys_brk"
eax_sysc_map[13] = "sys_rt_sigaction"
eax_sysc_map[14] = "sys_rt_sigprocmask"
eax_sysc_map[15] = "sys_rt_sigreturn"
eax_sysc_map[16] = "sys_ioctl"
eax_sysc_map[17] = "sys_pread64"
eax_sysc_map[18] = "sys_pwrite64"
eax_sysc_map[19] = "sys_readv"
eax_sysc_map[20] = "sys_writev"
eax_sysc_map[21] = "sys_access"
eax_sysc_map[22] = "sys_pipe"
eax_sysc_map[23] = "sys_select"
eax_sysc_map[24] = "sys_sched_yield"
eax_sysc_map[25] = "sys_mremap"
eax_sysc_map[26] = "sys_msync"
eax_sysc_map[27] = "sys_mincore"
eax_sysc_map[28] = "sys_madvise"
eax_sysc_map[29] = "sys_shmget"
eax_sysc_map[30] = "sys_shmat"
eax_sysc_map[31] = "sys_shmctl"
eax_sysc_map[32] = "sys_dup"
eax_sysc_map[33] = "sys_dup2"
eax_sysc_map[34] = "sys_pause"
eax_sysc_map[35] = "sys_nanosleep"
eax_sysc_map[36] = "sys_getitimer"
eax_sysc_map[37] = "sys_alarm"
eax_sysc_map[38] = "sys_setitimer"
eax_sysc_map[39] = "sys_getpid"
eax_sysc_map[40] = "sys_sendfile64"
eax_sysc_map[41] = "sys_socket"
eax_sysc_map[42] = "sys_connect"
eax_sysc_map[43] = "sys_accept"
eax_sysc_map[44] = "sys_sendto"
eax_sysc_map[45] = "sys_recvfrom"
eax_sysc_map[46] = "sys_sendmsg"
eax_sysc_map[47] = "sys_recvmsg"
eax_sysc_map[48] = "sys_shutdown"
eax_sysc_map[49] = "sys_bind"
eax_sysc_map[50] = "sys_listen"
eax_sysc_map[51] = "sys_getsockname"
eax_sysc_map[52] = "sys_getpeername"
eax_sysc_map[53] = "sys_socketpair"
eax_sysc_map[54] = "sys_setsockopt"
eax_sysc_map[55] = "sys_getsockopt"
eax_sysc_map[56] = "sys_clone"
eax_sysc_map[57] = "sys_fork"
eax_sysc_map[58] = "sys_vfork"
eax_sysc_map[59] = "sys_execve"
eax_sysc_map[60] = "sys_exit"
eax_sysc_map[61] = "sys_wait4"
eax_sysc_map[62] = "sys_kill"
eax_sysc_map[63] = "sys_newuname"
eax_sysc_map[64] = "sys_semget"
eax_sysc_map[65] = "sys_semop"
eax_sysc_map[66] = "sys_semctl"
eax_sysc_map[67] = "sys_shmdt"
eax_sysc_map[68] = "sys_msgget"
eax_sysc_map[69] = "sys_msgsnd"
eax_sysc_map[70] = "sys_msgrcv"
eax_sysc_map[71] = "sys_msgctl"
eax_sysc_map[72] = "sys_fcntl"
eax_sysc_map[73] = "sys_flock"
eax_sysc_map[74] = "sys_fsync"
eax_sysc_map[75] = "sys_fdatasync"
eax_sysc_map[76] = "sys_truncate"
eax_sysc_map[77] = "sys_ftruncate"
eax_sysc_map[78] = "sys_getdents"
eax_sysc_map[79] = "sys_getcwd"
eax_sysc_map[80] = "sys_chdir"
eax_sysc_map[81] = "sys_fchdir"
eax_sysc_map[82] = "sys_rename"
eax_sysc_map[83] = "sys_mkdir"
eax_sysc_map[84] = "sys_rmdir"
eax_sysc_map[85] = "sys_creat"
eax_sysc_map[86] = "sys_link"
eax_sysc_map[87] = "sys_unlink"
eax_sysc_map[88] = "sys_symlink"
eax_sysc_map[89] = "sys_readlink"
eax_sysc_map[90] = "sys_chmod"
eax_sysc_map[91] = "sys_fchmod"
eax_sysc_map[92] = "sys_chown"
eax_sysc_map[93] = "sys_fchown"
eax_sysc_map[94] = "sys_lchown"
eax_sysc_map[95] = "sys_umask"
eax_sysc_map[96] = "sys_gettimeofday"
eax_sysc_map[97] = "sys_getrlimit"
eax_sysc_map[98] = "sys_getrusage"
eax_sysc_map[99] = "sys_sysinfo"
eax_sysc_map[100] = "sys_times"
eax_sysc_map[101] = "sys_ptrace"
eax_sysc_map[102] = "sys_getuid"
eax_sysc_map[103] = "sys_syslog"
eax_sysc_map[104] = "sys_getgid"
eax_sysc_map[105] = "sys_setuid"
eax_sysc_map[106] = "sys_setgid"
eax_sysc_map[107] = "sys_geteuid"
eax_sysc_map[108] = "sys_getegid"
eax_sysc_map[109] = "sys_setpgid"
eax_sysc_map[110] = "sys_getppid"
eax_sysc_map[111] = "sys_getpgrp"
eax_sysc_map[112] = "sys_setsid"
eax_sysc_map[113] = "sys_setreuid"
eax_sysc_map[114] = "sys_setregid"
eax_sysc_map[115] = "sys_getgroups"
eax_sysc_map[116] = "sys_setgroups"
eax_sysc_map[117] = "sys_setresuid"
eax_sysc_map[118] = "sys_getresuid"
eax_sysc_map[119] = "sys_setresgid"
eax_sysc_map[120] = "sys_getresgid"
eax_sysc_map[121] = "sys_getpgid"
eax_sysc_map[122] = "sys_setfsuid"
eax_sysc_map[123] = "sys_setfsgid"
eax_sysc_map[124] = "sys_getsid"
eax_sysc_map[125] = "sys_capget"
eax_sysc_map[126] = "sys_capset"
eax_sysc_map[127] = "sys_rt_sigpending"
eax_sysc_map[128] = "sys_rt_sigtimedwait"
eax_sysc_map[129] = "sys_rt_sigqueueinfo"
eax_sysc_map[130] = "sys_rt_sigsuspend"
eax_sysc_map[131] = "sys_sigaltstack"
eax_sysc_map[132] = "sys_utime"
eax_sysc_map[133] = "sys_mknod"
eax_sysc_map[135] = "sys_personality"
eax_sysc_map[136] = "sys_ustat"
eax_sysc_map[137] = "sys_statfs"
eax_sysc_map[138] = "sys_fstatfs"
eax_sysc_map[139] = "sys_sysfs"
eax_sysc_map[140] = "sys_getpriority"
eax_sysc_map[141] = "sys_setpriority"
eax_sysc_map[142] = "sys_sched_setparam"
eax_sysc_map[143] = "sys_sched_getparam"
eax_sysc_map[144] = "sys_sched_setscheduler"
eax_sysc_map[145] = "sys_sched_getscheduler"
eax_sysc_map[146] = "sys_sched_get_priority_max"
eax_sysc_map[147] = "sys_sched_get_priority_min"
eax_sysc_map[148] = "sys_sched_rr_get_interval"
eax_sysc_map[149] = "sys_mlock"
eax_sysc_map[150] = "sys_munlock"
eax_sysc_map[151] = "sys_mlockall"
eax_sysc_map[152] = "sys_munlockall"
eax_sysc_map[153] = "sys_vhangup"
eax_sysc_map[154] = "sys_modify_ldt"
eax_sysc_map[155] = "sys_pivot_root"
eax_sysc_map[156] = "sys_sysctl"
eax_sysc_map[157] = "sys_prctl"
eax_sysc_map[158] = "sys_arch_prctl"
eax_sysc_map[159] = "sys_adjtimex"
eax_sysc_map[160] = "sys_setrlimit"
eax_sysc_map[161] = "sys_chroot"
eax_sysc_map[162] = "sys_sync"
eax_sysc_map[163] = "sys_acct"
eax_sysc_map[164] = "sys_settimeofday"
eax_sysc_map[165] = "sys_mount"
eax_sysc_map[166] = "sys_umount"
eax_sysc_map[167] = "sys_swapon"
eax_sysc_map[168] = "sys_swapoff"
eax_sysc_map[169] = "sys_reboot"
eax_sysc_map[170] = "sys_sethostname"
eax_sysc_map[171] = "sys_setdomainname"
eax_sysc_map[172] = "sys_iopl/ptregs"
eax_sysc_map[173] = "sys_ioperm"
eax_sysc_map[175] = "sys_init_module"
eax_sysc_map[176] = "sys_delete_module"
eax_sysc_map[179] = "sys_quotactl"
eax_sysc_map[186] = "sys_gettid"
eax_sysc_map[187] = "sys_readahead"
eax_sysc_map[188] = "sys_setxattr"
eax_sysc_map[189] = "sys_lsetxattr"
eax_sysc_map[190] = "sys_fsetxattr"
eax_sysc_map[191] = "sys_getxattr"
eax_sysc_map[192] = "sys_lgetxattr"
eax_sysc_map[193] = "sys_fgetxattr"
eax_sysc_map[194] = "sys_listxattr"
eax_sysc_map[195] = "sys_llistxattr"
eax_sysc_map[196] = "sys_flistxattr"
eax_sysc_map[197] = "sys_removexattr"
eax_sysc_map[198] = "sys_lremovexattr"
eax_sysc_map[199] = "sys_fremovexattr"
eax_sysc_map[200] = "sys_tkill"
eax_sysc_map[201] = "sys_time"
eax_sysc_map[202] = "sys_futex"
eax_sysc_map[203] = "sys_sched_setaffinity"
eax_sysc_map[204] = "sys_sched_getaffinity"
eax_sysc_map[206] = "sys_io_setup"
eax_sysc_map[207] = "sys_io_destroy"
eax_sysc_map[208] = "sys_io_getevents"
eax_sysc_map[209] = "sys_io_submit"
eax_sysc_map[210] = "sys_io_cancel"
eax_sysc_map[212] = "sys_lookup_dcookie"
eax_sysc_map[213] = "sys_epoll_create"
eax_sysc_map[216] = "sys_remap_file_pages"
eax_sysc_map[217] = "sys_getdents64"
eax_sysc_map[218] = "sys_set_tid_address"
eax_sysc_map[219] = "sys_restart_syscall"
eax_sysc_map[220] = "sys_semtimedop"
eax_sysc_map[221] = "sys_fadvise64"
eax_sysc_map[222] = "sys_timer_create"
eax_sysc_map[223] = "sys_timer_settime"
eax_sysc_map[224] = "sys_timer_gettime"
eax_sysc_map[225] = "sys_timer_getoverrun"
eax_sysc_map[226] = "sys_timer_delete"
eax_sysc_map[227] = "sys_clock_settime"
eax_sysc_map[228] = "sys_clock_gettime"
eax_sysc_map[229] = "sys_clock_getres"
eax_sysc_map[230] = "sys_clock_nanosleep"
eax_sysc_map[231] = "sys_exit_group"
eax_sysc_map[232] = "sys_epoll_wait"
eax_sysc_map[233] = "sys_epoll_ctl"
eax_sysc_map[234] = "sys_tgkill"
eax_sysc_map[235] = "sys_utimes"
eax_sysc_map[237] = "sys_mbind"
eax_sysc_map[238] = "sys_set_mempolicy"
eax_sysc_map[239] = "sys_get_mempolicy"
eax_sysc_map[240] = "sys_mq_open"
eax_sysc_map[241] = "sys_mq_unlink"
eax_sysc_map[242] = "sys_mq_timedsend"
eax_sysc_map[243] = "sys_mq_timedreceive"
eax_sysc_map[244] = "sys_mq_notify"
eax_sysc_map[245] = "sys_mq_getsetattr"
eax_sysc_map[246] = "sys_kexec_load"
eax_sysc_map[247] = "sys_waitid"
eax_sysc_map[248] = "sys_add_key"
eax_sysc_map[249] = "sys_request_key"
eax_sysc_map[250] = "sys_keyctl"
eax_sysc_map[251] = "sys_ioprio_set"
eax_sysc_map[252] = "sys_ioprio_get"
eax_sysc_map[253] = "sys_inotify_init"
eax_sysc_map[254] = "sys_inotify_add_watch"
eax_sysc_map[255] = "sys_inotify_rm_watch"
eax_sysc_map[256] = "sys_migrate_pages"
eax_sysc_map[257] = "sys_openat"
eax_sysc_map[258] = "sys_mkdirat"
eax_sysc_map[259] = "sys_mknodat"
eax_sysc_map[260] = "sys_fchownat"
eax_sysc_map[261] = "sys_futimesat"
eax_sysc_map[262] = "sys_newfstatat"
eax_sysc_map[263] = "sys_unlinkat"
eax_sysc_map[264] = "sys_renameat"
eax_sysc_map[265] = "sys_linkat"
eax_sysc_map[266] = "sys_symlinkat"
eax_sysc_map[267] = "sys_readlinkat"
eax_sysc_map[268] = "sys_fchmodat"
eax_sysc_map[269] = "sys_faccessat"
eax_sysc_map[270] = "sys_pselect6"
eax_sysc_map[271] = "sys_ppoll"
eax_sysc_map[272] = "sys_unshare"
eax_sysc_map[273] = "sys_set_robust_list"
eax_sysc_map[274] = "sys_get_robust_list"
eax_sysc_map[275] = "sys_splice"
eax_sysc_map[276] = "sys_tee"
eax_sysc_map[277] = "sys_sync_file_range"
eax_sysc_map[278] = "sys_vmsplice"
eax_sysc_map[279] = "sys_move_pages"
eax_sysc_map[280] = "sys_utimensat"
eax_sysc_map[281] = "sys_epoll_pwait"
eax_sysc_map[282] = "sys_signalfd"
eax_sysc_map[283] = "sys_timerfd_create"
eax_sysc_map[284] = "sys_eventfd"
eax_sysc_map[285] = "sys_fallocate"
eax_sysc_map[286] = "sys_timerfd_settime"
eax_sysc_map[287] = "sys_timerfd_gettime"
eax_sysc_map[288] = "sys_accept4"
eax_sysc_map[289] = "sys_signalfd4"
eax_sysc_map[290] = "sys_eventfd2"
eax_sysc_map[291] = "sys_epoll_create1"
eax_sysc_map[292] = "sys_dup3"
eax_sysc_map[293] = "sys_pipe2"
eax_sysc_map[294] = "sys_inotify_init1"
eax_sysc_map[295] = "sys_preadv"
eax_sysc_map[296] = "sys_pwritev"
eax_sysc_map[297] = "sys_rt_tgsigqueueinfo"
eax_sysc_map[298] = "sys_perf_event_open"
eax_sysc_map[299] = "sys_recvmmsg"
eax_sysc_map[300] = "sys_fanotify_init"
eax_sysc_map[301] = "sys_fanotify_mark"
eax_sysc_map[302] = "sys_prlimit64"
eax_sysc_map[303] = "sys_name_to_handle_at"
eax_sysc_map[304] = "sys_open_by_handle_at"
eax_sysc_map[305] = "sys_clock_adjtime"
eax_sysc_map[306] = "sys_syncfs"
eax_sysc_map[307] = "sys_sendmmsg"
eax_sysc_map[308] = "sys_setns"
eax_sysc_map[309] = "sys_getcpu"
eax_sysc_map[310] = "sys_process_vm_readv"
eax_sysc_map[311] = "sys_process_vm_writev"
eax_sysc_map[312] = "sys_kcmp"
eax_sysc_map[313] = "sys_finit_module"
eax_sysc_map[314] = "sys_sched_setattr"
eax_sysc_map[315] = "sys_sched_getattr"
eax_sysc_map[316] = "sys_renameat2"
eax_sysc_map[317] = "sys_seccomp"
eax_sysc_map[318] = "sys_getrandom"
eax_sysc_map[319] = "sys_memfd_create"
eax_sysc_map[320] = "sys_kexec_file_load"
eax_sysc_map[321] = "sys_bpf"
eax_sysc_map[322] = "sys_execveat/ptregs"
eax_sysc_map[323] = "sys_userfaultfd"
eax_sysc_map[324] = "sys_membarrier"
eax_sysc_map[325] = "sys_mlock2"
eax_sysc_map[326] = "sys_copy_file_range"
eax_sysc_map[327] = "sys_preadv2"
eax_sysc_map[328] = "sys_pwritev2"
eax_sysc_map[329] = "sys_pkey_mprotect"
eax_sysc_map[330] = "sys_pkey_alloc"
eax_sysc_map[331] = "sys_pkey_free"
eax_sysc_map[332] = "sys_statx"
eax_sysc_map[MaFeatures.NOCLASS] = "none"

def id2func(sc_key, shift=0x120, mult=8):
    if sc_key == MaFeatures.NOCLASS:
        return 'none'
    key = (sc_key - shift) / mult
    try:
        return eax_sysc_map[key].lower()
    except KeyError:
        return 'invalid'

if __name__ == "__main__":
    ma_dumps = [
            "/src/workspace/mempattern/memaccess_dumps/r0_pw.dump",
            "/src/workspace/mempattern/memaccess_dumps/r1_pw.dump"
            ]
    annots = [
            "/src/workspace/mempattern/memaccess_dumps/r0_pw.dump",
            "/src/workspace/mempattern/memaccess_dumps/r1_pw.dump"
            ]


    maf = MaFeatures()
    maf.add_data(ma_dumps[0])
    #maf.add_data(ma_dumps[1])
    b2ma_df = maf.get_block_features()
    print(maf.df.head())
    print(b2ma_df.head())
