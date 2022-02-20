import os
import random
import math
import pickle
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import scipy.stats as stats
from scipy.stats import norm
import multiprocessing


class MaFeatures(object):
    NOCLASS = 500
    COLUMNS_DATA =  ['KEY', 'MAC']
    COLUMNS_ANOT =  ['KEY', 'IDX_0']
    def __init__(self):
        self.df = pd.DataFrame(columns = MaFeatures.COLUMNS_DATA)
        self.df['KEY'] = self.df['KEY'].astype('uint64')
        self.df['MAC'] = self.df['MAC'].astype('uint64')
        self.annot_df = pd.DataFrame(columns = MaFeatures.COLUMNS_ANOT)
        self.annot_df['KEY'] = self.annot_df['KEY'].astype('uint64')
        self.annot_df['IDX_0'] = self.annot_df['IDX_0'].astype('uint64')

    def __filepos_to_arrayindex(self, filepos):
        return (int) (filepos / 16)

    def add_ma(self, ma_path, info_path):
        print('[+] Loading Memory Access File %s' % ma_path)
        dump_size = os.stat(ma_path).st_size # in bytes
        n_cols = 2
        n_rows = int(dump_size / (n_cols * 8))

        new_data = np.memmap(
            ma_path,
            dtype=np.uint64,
            mode='c',
            shape=(n_rows, n_cols))

        start_idx = len(self.df.index)
        stop_idx = start_idx + n_rows
        print('[+] new index %d-%d' % (start_idx, stop_idx))
        new_df = pd.DataFrame(
                data=new_data,
                columns=MaFeatures.COLUMNS_DATA,
                index=range(start_idx, stop_idx),
                dtype=np.uint64)


        info = pickle.load(open(info_path, "rb"))
        kstext = np.repeat(info['kstext'], stop_idx - start_idx).astype('uint64')
        new_df['KEY'] = new_df['KEY'] - kstext

        new_df['KEY'] = new_df['KEY'].astype('uint64')
        new_df['MAC'] = new_df['MAC'].astype('uint64')

        return new_df

    def add_annot(self, annot_path, typ):
        print("[+] Loading Annotation File: %s" % annot_path)
        annots = json.load(open(annot_path, 'r'))

        new_annot_data = []
        for a in annots:
            new_annot_data.append({
                'KEY': a['cpuinfo']['EAX'],
                'IDX_0': self.__filepos_to_arrayindex(a[typ])
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

        new_annot_df['KEY'] = new_annot_df['KEY'].astype('uint8')
        new_annot_df['IDX_0'] = new_annot_df['IDX_0'].astype('uint32')
        return new_annot_df


    def add_data(self, ma_path, info_path, annot_path=None, typ=None):
        new_df = self.add_ma(ma_path, info_path)
        self.df = self.df.append(new_df)
        if annot_path is not None:
            new_annot_df = self.add_annot(annot_path, typ)
            self.annot_df = self.annot_df.append(new_annot_df)

    def get_block_features(self):
        new_df = self.df.copy()
        new_df['sentinel'] = (new_df['KEY'] != new_df['KEY'].shift(1)).astype(int).cumsum()
        new_df = new_df.groupby(['sentinel', 'KEY'])['MAC'].apply(list)
        return new_df

    def get_annot_features_step_gap(self, gap, step, noclass_perc=2.5, n=20):
        if step <= gap:
            print("Ivalid step/gap value: step=%d, gap=%d\n", step, gap)
            return None

        new_df = self.annot_df.copy()
        n_noclass = int(new_df.index[-1] * noclass_perc)
        new_df = new_df.append(
            pd.DataFrame([
                {'KEY': 500, 'IDX_0': i} for i in np.random.randint(self.df.index[-1], size=(n_noclass))],
                index=range(new_df.index[-1] +1, new_df.index[-1] +1 + n_noclass)))

        idx = 0
        for i in range(0, n):
            if (idx % step) == 0:
                idx += gap
            new_df['IDX_%d' % i] = new_df['IDX_0'] + idx
            idx += 1

        for i in range(0, n):
            new_df['MA_%d' % i] = self.df.iloc[new_df['IDX_%d' % i]]['MAC'].values

        for i in range(1, n):
            new_df['MAD_%d' % i] = new_df['MA_%d' % i] - new_df['MA_%d' % (i - 1)]

        return new_df

    def get_annot_features4(self, sc_offsets, start, n, n_items=1000, n_step=2):
        r_df = pd.DataFrame(columns=['KEY', 'IDX_0'], dtype=np.uint64)
        stop = start + n_items * n_step
        if not hasattr(self, "sc_indices"):
            self.sc_indices = self.df[self.df['KEY'].isin(sc_offsets)].index

        sci_marker = 0
        while self.sc_indices[sci_marker] < start:
            sci_marker += 1

        df_idx = self.df.iloc[start:stop:n_step].index
        r_df = pd.DataFrame(columns=['KEY', 'IDX_0'], data={'KEY': 500, 'IDX_0': df_idx}, index=df_idx)
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



    def get_annot_features3(self, sc_offsets, noclass_perc=2.5, n=20):
        r_df = pd.DataFrame(columns=['KEY', 'IDX_0'], dtype=np.uint64)
        starti = 0
        stopi = 0
        #sc_indices = set()
        sc_indices = set(range(len(self.df.index)))
        for sc_off in sc_offsets:
            #print('[+] adding sc offset %x' % sc_off)
            #data = self.df[self.df['KEY'] == sc_off].index
            data = np.flatnonzero(self.df['KEY'] == sc_off)
            #data = list(filter(lambda t: t<len(self.df.index) - 1, data))
            for idx in data:
                sc_indices -= set(range(max(0, idx - n), idx + n))
            #    sc_indices |= set(range(max(0, idx - n), idx + n))
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
                #{'KEY': 500, 'IDX_0': i} for i in np.random.randint(len(self.df.index) - n, size=(n_noclass))],
                {'KEY': 500, 'IDX_0': i} for i in random.sample(sc_indices, n_noclass)],
                index=range(len(r_df.index), len(r_df.index) + n_noclass))
        #drop_indices = new_df[new_df['IDX_0'].isin(sc_indices)].index
        #if not drop_indices.empty:
        #    print('[!] dropping %d of %d indices for noclass' % (len(drop_indices), len(new_df.index)))
        #    new_df = new_df.drop(drop_indices)
        r_df = r_df.append(new_df, sort=False)

        r_df['KEY'] = r_df['KEY'].astype('uint64')
        r_df['IDX_0'] = r_df['IDX_0'].astype('uint64')

        for i in range(1, n):
            r_df['IDX_%d' % i] = r_df['IDX_0'] + i
            r_df['IDX_%d' % i] = r_df['IDX_%d' % i].astype('uint64')

        for i in range(0, n):
            r_df['MA_%d' % i] = self.df.iloc[r_df['IDX_%d' % i]]['MAC'].astype('uint64').values
            r_df['IP_%d' % i] = self.df.iloc[r_df['IDX_%d' % i]]['KEY'].astype('uint64').values

        for i in range(1, n):
            r_df['MAD_%d' % i] = r_df['MA_%d' % i].astype('int64') - r_df['MA_%d' % (i - 1)].astype('int64')

        return r_df


    def get_annot_features2(self, noclass_perc=2.5, n=20):
        new_df = self.annot_df.copy()
        n_noclass = int(new_df.index[-1] * noclass_perc)
        new_df = new_df.append(
            pd.DataFrame([
                {'KEY': 500, 'IDX_0': i} for i in np.random.randint(self.df.index[-1], size=(n_noclass))],
                index=range(new_df.index[-1] +1, new_df.index[-1] +1 + n_noclass)), sort=False)
        for i in range(1, n):
            new_df['IDX_%d' % i] = new_df['IDX_0'] + i

        for i in range(0, n):
            new_df['MA_%d' % i] = self.df.iloc[new_df['IDX_%d' % i]]['MAC'].values

        for i in range(1, n):
            new_df['MAD_%d' % i] = new_df['MA_%d' % i] - new_df['MA_%d' % (i - 1)]

        return new_df

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
        return pd.DataFrame(data=fdata, columns=c)


rid = "r10"
outdir = "/src/workspace2/mempattern/out"
benchdir = "/src/workspace2/mempattern/out/%s" % rid

indir = "/src/workspace2/mempattern/memaccess_dumps/%s_ltp_kaslr_split_1G/" % rid
info = "/src/workspace/mempattern/recordings/%s_ltp_kaslr.info" % rid
if not os.path.exists(benchdir):
    os.mkdir(benchdir)

sc_offs = list(map(lambda t: int(t, 16), open("/src/workspace/mempattern/out/sc_offs", "r").readlines()))
sc_names = list(map(lambda t: t[1:-3], open("/src/workspace/mempattern/out/sc_names", "r").readlines()))
sc_o2n = {500: 'none'}
sc_n2o = {'none': 500}
for o, n in zip(sc_offs, sc_names):
    sc_o2n[o] = n.lower()
    sc_n2o[n.lower()] = o

mas = [os.path.join(indir, f) for f in \
       os.listdir(indir) \
       if os.path.isfile(os.path.join(indir, f))]

loaded_model = pickle.load(open(os.path.join(outdir, "model.sav"), 'rb'))
sc_sel_names = pickle.load(open(os.path.join(outdir, "model.names"), 'rb'))

sc_sel_offsets = [x[1] for x in sc_n2o.items() if x[0] in sc_sel_names]
sc_sel_offsets


n = 400
n_items = 1024
n_step = 1
def ev(ma):
    print(ma)

    maf = MaFeatures()
    maf.add_data(ma, info)

    start = 0
    ldf = len(maf.df.index) - n * n_step
    with open(os.path.join(benchdir, "model_test_%s.log" % os.path.basename(ma)), "w") as fd:
        while start < ldf:
            #print("%.2f\r" % (start/ldf), end="")
            #print(start)
            test_f, start = maf.get_annot_features4(sc_sel_offsets, start, n, n_items=n_items, n_step=n_step)
            f_headers = list(filter(lambda x: x.startswith('MAD_'), test_f.columns.values))
            t_header = ['KEY']
            #test_f.groupby(t_header)['KEY', 'IDX_0'].count().to_csv(
            #    os.path.join(outdir, "model_keys_%d_%s.csv" %(start, os.path.basename(ma)))
            #)
            test_x = test_f[f_headers].values
            test_y = test_f[t_header].values
            predictions = loaded_model.predict(test_x)
            acs = accuracy_score(test_y, predictions)
            if acs != 1.0:
                names = []
                for x in sorted(set(test_f['KEY'].values)):
                    names.append(sc_o2n[x])
                pickle.dump(names,
                            open(os.path.join(benchdir, "model_test_%s_%d.names" % (os.path.basename(ma), start)),
                            "wb")
                           )
                pickle.dump(confusion_matrix(test_y, predictions),
                            open(os.path.join(benchdir, "model_test_%s_%d.cmat" % (os.path.basename(ma), start)),
                            "wb")
                           )
            #print("Test Accuracy  :: ", acs)
            fd.write('=' * 80)
            fd.write('\n')
            fd.write("SIDX %d\n" % start)
            fd.write("Keys: ")
            scs, counts = np.unique(test_y, return_counts=True)
            fd.write(", ".join(map(lambda t: "%x (%d)" % (t[0], t[1]), zip(scs, counts))))
            fd.write("\n")
            fd.write("Acc: %f\n" % acs)

            false_pos = 0
            false_neg = 0
            for tst, prd in zip(test_y, predictions):
                if tst[0] != 500 and prd == 500:
                    false_neg += 1
                elif tst[0] == 500 and prd != 500:
                    false_pos += 1

            fd.write("FP: %d/%d\n" % (false_pos, len(test_y)))
            fd.write("FN: %d/%d\n" % (false_neg, len(test_y)))
            fd.flush()

print("starting MP")
with multiprocessing.Pool(4) as p:
        p.map(ev, mas)
        print("done")
