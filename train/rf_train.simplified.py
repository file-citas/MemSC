# %%
import contextlib
import pathlib
import hashlib
import json
import sys
import tempfile
import urllib
import os
import collections
import h5py
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
from numpy import inf
import bz2
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
import seaborn as sns
sns.set(font_scale= 1.0)
import gc
import time
from skimpy import skim
from pycaret.utils import *
from pycaret.classification import *
import natsort
from natsort import *
import os
import re
import glob
print('pycaret',version())

# %%
def not_missing_or_empty(fpath):
    return os.path.exists(fpath) and os.path.isfile(fpath) and os.path.getsize(fpath) > 0

# %%
def sort_dict(d):
    #sort dict according alphabetically to key
    sorted_key = sorted(d.items(), key = lambda kv: kv[0])
    return dict(sorted_key)

# %%
plt.ioff()
config_path='./config.json'
config_dict={} 
if not_missing_or_empty(config_path):
    print(f"Configuration is taken from {config_path} ")
    with open('./config.json','r') as f:
        config_dict=json.load(f)
else:
    print(f"Configuration is taken from program defaults")
    #pycaret parameters
    config_dict['target']='KEY'
    config_dict['session_id']=125
    config_dict['verbose']=True
    config_dict['normalize']=True
    config_dict['normalize_method']='robust'
    config_dict['fix_imbalance']=False
    config_dict['fix_imbalance_method']=None
    config_dict['feature_selection']=False
    config_dict['feature_selection_method']='boruta'
    config_dict['ignore_low_variance']=True
    config_dict['remove_multicollinearity']=False
    config_dict['remove_outliers']=False
    config_dict['create_clusters']=False
    config_dict['n_jobs']=10
    config_dict['use_gpu']=False
    config_dict['log_experiment']=False #When set to True, all metrics and parameters are logged on MLFlow server.
    config_dict['experiment_name']=None #Name of experiment for logging. When set to None, ‘clf’ is by default used as alias for the experiment name.
    config_dict['log_plots']=False #When set to True, specific plots are logged in MLflow as a png file. By default, it is set to False.
    config_dict['log_profile']=False #When set to True, data profile is also logged on MLflow as a html file. By default, it is set to False.
    config_dict['log_data']=False #When set to True, train and test dataset are logged as csv.
    config_dict['fold_strategy']='stratifiedkfold' #'kfold', 'groupkfold', 'timeseries'
    config_dict['data_split_stratify']=False
    config_dict['fold']=10
    #
    config_dict['tune_baseline']=True 
    tune_dict={}
    Metric='Accuracy'
    config_dict['Metric']=Metric
    tune_dict['default']=['50',''+Metric+'','scikit-learn','random','asha',True]
    tune_dict['scikit_optimize']=['50',''+Metric+'','scikit-optimize','bayesian','asha',True]
    tune_dict['sklearn_bayesian']=['50',''+Metric+'','tune-sklearn','bayesian','asha',True]
    tune_dict['sklearn_hyperopt']=['50',''+Metric+'','tune-sklearn','hyperopt','asha',True]
    tune_dict['sklearn_optuna']=['50',''+Metric+'','tune-sklearn','optuna','asha',True]
    tune_dict['optuna']=['50',''+Metric+'','optuna','tpe','asha',True]
    config_dict['tune_dict']=tune_dict
    #
    config_dict['plot_baseline']=True 
    plot_dict={}
    plot_dict['error']=['Prediction Error','Class-Prediction-Error','.png',True] 
    plot_dict['boundary']=['Decision Boundary','Decision-Boundary','.png',True] 
    plot_dict['rfe']=['Recursive Feature Selection','Recursive-Feature-Selection','.png',False] 
    plot_dict['manifold']=['Manifold Learning','Manifold-Learning','.png',False] 
    plot_dict['calibration']=['Calibration Curve','Calibration-Curve','.png',False] 
    plot_dict['parameter']=['Model Hyperparameter','Model Hyperparameter','.png',False]
    plot_dict['lift']=['Lift Curve','Lift-Curve','.png',False]
    plot_dict['gain']=['Gain Chart','Gain-Chart','.png',False]
    plot_dict['tree']=['Decision Tree','Decision-Tree','.png',False]
    plot_dict['ks']=['KS Statistic Plot','KS-Statistic-Plot','.png',False]
    plot_dict['class_report']=['Class Report','Classification-Report','.png',True]
    plot_dict['auc']=['AUC','Area-Under-the-Curve','.png',True]
    plot_dict['threshold']=['Discrimination Threshold','Discrimination-Threshold','.png',False]
    plot_dict['confusion_matrix']=['Confusion Matrix','Confusion-Matrix','.png',True] 
    plot_dict['feature']=['Feature Importance','Feature-Importance','.png',True]
    plot_dict['feature_all']=['Feature Importance (All)','Feature-Importance-All','.png',True]
    plot_dict['learning']=['Learning Curve','Learning-Curve','.png',True] 
    plot_dict['pr']=['Precision Recall','Precision-Recall-Curve','.png',True] 
    plot_dict['vc']=['Validation Curve','Validation-Curve','.png',True]
    config_dict['plot_dict']=plot_dict
    #
    config_dict['cm_cmap']='Blues'
    #
    config_dict['eda_plots']=False 
    config_dict['box_plot']=True
    config_dict['violin_plot']=True
    config_dict['count_plot']=True
    config_dict['profile_plot']=True
    config_dict['mean_profile_plot']=True
    config_dict['feature_correlation_plot']=True
    config_dict['set_dark_cs']=False
    config_dict['umap_plot']=True
    #
    config_dict['LS_paper']=244
    #
    config_dict['model_name']='Random_Forrest'
    config_dict['model']='rf'
    config_dict['experiment_name_suffix']='auto'
    #
    config_dict['use_existing_feature_rejections']=True
    config_dict['train_test_split']=False
    config_dict['do_not_drop_in_group_duplicates_below_fill_up']=True
    #
    config_dict['fill_up']=100
    config_dict['exact_fill_up']=False
    config_dict['size_limit']=10000
    config_dict['size_cut']=0
    config_dict['n_trials']=0
    #
    config_dict['log2_transform']=False 
    config_dict['intergroup_drop']='' #value range: '','first','last',False  #(ad 'Most_Frequent')
    config_dict['ingroup_drop']='' #value range: '','first','last',False
    config_dict['size_limit_ingroup_drop']=config_dict['fill_up']
    #
    config_dict['store_train_and_test_seperately']=False
    config_dict['stop_at_missing_nonsc']=True
    config_dict['syscall_path']='./'
    config_dict['h5_path']='./'
    #
    config_dict['l_channel']=['pr','pw','ma']  #channel: pr, pw, ma
    config_dict['l_blur']=natsorted([0,4,12]) #mask: 0=unmasked,4=masked,12=DMA
    config_dict['l_nfeat']=natsorted([512,256,192,128,96,64,48,32,24,16,8])
    #
    eval_default='./eval_file.h5'
    test_file_dict={} 
    for channel in config_dict['l_channel']:
        for blur in config_dict['l_blur']:
            bv=str(channel)+'_'+str(blur)
            test_file_dict[bv] =eval_default
    config_dict['test_file_dict']=test_file_dict  
    #
    load_model_default='./final.model.pkl'
    load_model_dict={} 
    for channel in config_dict['l_channel']:
        for nfeat in config_dict['l_nfeat']:
            for blur in config_dict['l_blur']:
                bv=str(channel)+'_'+str(nfeat)+'_'+str(blur)
                load_model_dict[bv]='./final.model.pkl'
    config_dict['load_model_dict']=load_model_dict  
    #
    with open(config_path, 'w') as file:
        json.dump(config_dict,file)
        print(f"{config_path} with program defaults has been generated")
#
config_dict=sort_dict(config_dict)
config_hash=hashlib.md5(open(config_path,'rb').read()).hexdigest()
print(f"Hash for {config_path} is {config_hash}")
for f in glob.iglob("./config.sorted.*"):
    if os.path.exists(f):
        os.remove(f)
with open("./config.sorted."+config_hash,'w') as file:
    json.dump(config_dict,file)
    print(f"{file} with alphabetically sorted program defaults has been generated:")
    print(json.dumps(config_dict, sort_keys=True,indent=4, separators=(',', ': ')))
#transform dictionary key, value to variable assignment key=value
for key in config_dict:
    if isinstance(config_dict[key],str):
        exec(key+"="+"'"+config_dict[key]+"'")
    else:
        exec(key+"="+str(config_dict[key]))
try:
    experiment_name_suffix  # does a exist in the current namespace
except NameError:
    experiment_name_suffix = 'auto' # nope
if experiment_name_suffix=="auto":
    experiment_name_suffix="."+config_hash
print(f"Experiment suffix for this calculation is {experiment_name_suffix}")
print(f"Current working directory is {os.path.abspath(os.path.curdir)}")

# %%
#logtransform an arbitrary matrix with non-positive entries
#Calculate log to basis b of a Matrix allowing entries <= 0
def logb(A,b=2):
    lb=np.log10(b)
    old_settings=np.seterr(divide='ignore', invalid='ignore')
    P=np.log10(A)
    P[P==-inf]=0
    np.nan_to_num(P,copy=False)
    if np.any(A<0):
        N=np.log10(-A)
        N[N==-inf]=0
        np.nan_to_num(N,copy=False)
        _=np.seterr(**old_settings)
        A=(P-N)/lb
    else:
        A=P/lb
    _=np.seterr(**old_settings)
    return A
def label_set(df,na):
    Y=df['KEY'].to_numpy(copy=True)
    S=natsorted(sorted(set(Y)))
    print(len(S),"train labels",na,"found:")
    print(S)
    return S
def calc_weights(df,na):
    print("sample_weights",na+":")
    Y=pd.DataFrame(df['KEY'])
    Y.reset_index(drop=True, inplace=True)
    W=sklearn.utils.class_weight.compute_sample_weight("balanced",Y,indices=None)
    W=pd.DataFrame({'W': W})
    W=pd.concat([Y,W],axis=1)
    W.drop_duplicates(inplace=True,ignore_index=True)
    W.sort_values(by=['KEY'],ascending=True,inplace=True)
    W.to_csv(experiment_name+'.sample_weights.csv',index=0)
    print('extrema of train weights:')
    M=W['W'].max()
    m=W['W'].min()
    print(m,M)
    bins,_=W.shape
    binrange="("+str(int(m))+","+str(int(M)+1)+")"
    W.drop('KEY',axis=1).describe().T.style.bar(subset=['mean'], color='#205ff2')\
                        .background_gradient(subset=['std'], cmap='Reds')\
                        .background_gradient(subset=['50%'], cmap='coolwarm')
    plt.rcParams["figure.figsize"] = (10, 10)
    sns.histplot(data=W['W'],bins=bins,binrange=(int(m),int(M)+1),kde=True,element="step",stat='probability')
    plt.plot()
    plt.savefig(experiment_name+'.LABEL_WEIGHTS.png')
    plt.close()
    return dict(zip(W['KEY'],W['W']))
def compare_label_sets(A,B,na,nb):
   """compare two label sets
   Args:
       A ([type]): [description]
       B ([type]): [description]
       na ([type]): [description]
       nb ([type]): [description]
   """   
   incons=0
   if(len(A)>len(B)):
      print('There are more '+na+' labels than '+nb+' labels')
      incons=1
   if(len(A)<len(B)):
      print('There are less '+na+' labels than '+nb+' labels')
   if(len(set(A)-set(B))>0):
      print('The following '+na+' labels are not in the '+nb+' labels:')
      print(natsorted(sorted(set(A)-set(B))))
      incons=1
   if(len(set(B)-set(A))>0):
      print('The following '+nb+' labels are not in the '+na+' label set:')
      print(natsorted(sorted(set(B)-set(A))))
   if(len(set(B)&set(A))>0):
      print('The following labels are in both label sets:')
      print(natsorted(sorted(set(B)&set(A))))   
   if(incons==0):
       print("The "+na+' label set is contained in the '+nb+' label set')
   return
def make_syscall_dicts(D):
    """Generate syscall dictionary from syscall file
    Args:
        D (pandas df): NR,SC csv-file
    """ 
    nr2sc=pd.read_csv(D)
    nr2sc.drop_duplicates(inplace=True)
    nr2sc.reset_index(drop=True, inplace=True)
    return dict(zip(nr2sc['NR'],nr2sc['SC'])),dict(zip(nr2sc['SC'],nr2sc['NR']))
def Size_Profile(df):
    size_profile=df["KEY"].value_counts().to_frame()
    size_profile.reset_index(level=0,inplace=True)
    size_profile.columns=['ID','COUNT']
    size_profile=pd.concat([size_profile['ID'].replace(nr2sc),size_profile],axis=1)
    size_profile.columns=['NAME','ID','COUNT']
    size_profile.sort_values(by=['COUNT'],ascending=True,inplace=True)
    return size_profile

# %%
with open(syscall_path+'syscalls.csv') as f:
    nr2sc,sc2nr=make_syscall_dicts(f)
    with open('NR2SC.json', 'w') as file:
        json.dump(nr2sc, file)
    with open('SC2NR.json', 'w') as file:
        json.dump(sc2nr, file)
#start writing eval_dict
#add <syscall nr> as column to eval_dict
eval_dict={} 
eval_dict_columns=['NR']
for key in sc2nr.keys():
    eval_dict[key]=[sc2nr[key]]
nr_set=set(nr2sc.keys())
sc_set=set(sc2nr.keys())
max_nr_sc=len(nr2sc)
max_nr_sc_paper=314
print('Size of current syscall dictionary         :',max_nr_sc)
print('Size of         syscall dictionary in paper:',max_nr_sc_paper)
print("")
with open(syscall_path+'SC-Missing.txt') as f:
    missing_sc = [line.rstrip() for line in f]
print(str(len(missing_sc))+' sycalls tagged as missing in paper:')
#add <mentioned as missing in paper> column to eval_dict
sc_set_not_covered=set(missing_sc)
eval_dict_columns.append('P_MISS')
for key in eval_dict.keys():
    if key in missing_sc:
        sc_set_not_covered.remove(key)
        eval_dict[key].append(True)
    else:
        eval_dict[key].append(False)
if sc_set_not_covered != set():
    print('Warning: The following keys declared as missing in paper are not covered by basic key list:')
    print(sc_set_not_covered)
print(natsorted(missing_sc))
missing_nr=[sc2nr[na] for na in missing_sc]
print('Corresponding NR list')
print(missing_nr)
print("")
with open(syscall_path+'SC-Excluded.txt') as f:
    excluded_sc = [line.rstrip() for line in f]
excluded_sc_not_covered=set(excluded_sc)
#add <scs excluded in paper> as column to eval_dict
eval_dict_columns.append('P_EXCL')
for key in eval_dict.keys():
    if key in excluded_sc:
        excluded_sc_not_covered.remove(key)
        eval_dict[key].append(True)
    else:
        eval_dict[key].append(False)
if excluded_sc_not_covered != set():
    print('Warning: The following keys declared as excluded from paper are not covered by basic key list:')
    print(excluded_sc_not_covered)
print(str(len(excluded_sc))+' sycalls tagged as excluded in paper:')
print(natsorted(excluded_sc))
excluded_nr=[sc2nr[na] for na in excluded_sc]
print('Corresponding NR list')
print(excluded_nr)
print("")
excluded=natsorted(set(excluded_nr)|set(missing_nr))
print("In total, the following "+str(len(excluded))+" syscalls are tagged as \"to be skipped\" in the paper:")
print(excluded)
print('')
#
try:
    with open(syscall_path+'SC-Explicitly-Mentioned') as f:
        explicitly_mentioned_sc = [line.rstrip() for line in f]
except IOError:
        print(IOError)
if len(explicitly_mentioned_sc)==0:
    print('Error:',syscall_path+'SC-Explicitly-Mentioned','is empty')
explicitly_mentioned_sc_not_covered=set(explicitly_mentioned_sc)
#add explicitly mentioned in paper column to eval_dict
eval_dict_columns.append('P_Ment')
for key in eval_dict.keys():
    if key in explicitly_mentioned_sc:
        explicitly_mentioned_sc_not_covered.remove(key)
        eval_dict[key].append(True)
    else:
        eval_dict[key].append(False)
if explicitly_mentioned_sc_not_covered != set():
    print('Warning: The following keys explicitly mentioned in paper are not covered by basic key list:')
    print(explicitly_mentioned_sc_not_covered)
print(str(len(explicitly_mentioned_sc))+' sycalls explicitly shown in paper pictures:')
print(natsorted(explicitly_mentioned_sc))
explicitly_mentioned_nr=[sc2nr[na] for na in explicitly_mentioned_sc]
print('Corresponding NR list')
print(explicitly_mentioned_nr)
print("")

# %%
def Balance_Groups(df,fill_up):
    print(f"Shape at start of group balancing {df.shape}")
    Shape_Book(df.shape[0],df.shape[1],shape_count,"start group balancing")
    groupr=df.groupby('KEY')
    #number of groups
    C=len(groupr)
#    for key,group in groupr:
#        C+=1
    if ingroup_drop != '': 
        li=[]
        c=0
        for key,group in groupr:
            groupn=group.drop_duplicates(subset=None,ignore_index=True,keep=ingroup_drop)
            if groupn.shape[0]<groupn.shape[0] and groupn.shape[0] > size_limit_ingroup_drop:
                c+=1
                li.append(groupn)
            else:
                li.append(group)
        if c>0:
            df=pd.concat(li,axis=0,ignore_index=True).sort_values(by=['KEY'],axis=0,ascending=True,ignore_index=True)
            print(f"Duplicates have been dropped from {c}/{C} groups")
            print(f"Shape after in-group drop of key duplicates {df.shape}")
            Shape_Book(df.shape[0],df.shape[1],shape_count,"after in-group drop of key duplicates")
        else:
            print("No ingroup duplicates found")
    else:
        print("No (potential) ingroup duplicates dropped")
    #
    size_profile=Size_Profile(df)
    NL=size_profile.nlargest(2,'COUNT')
    lg_na=NL.iloc[0,0]
    lg_co=NL.iloc[0,2]
    nlg_na=NL.iloc[1,0]
    nlg_co=NL.iloc[1,2]
    li=[]
    lg_co=min(size_limit,lg_co)
    nlg_co=min(size_limit,nlg_co)
    fill_up_list=[]
    down=0
    up=0
    for key,group in groupr:
        s=group.shape[0]
        t=0
        if s > size_limit:
            t=1
            down+=1
            group=group.sample(n=size_limit,axis=0,ignore_index=True,random_state=42)
            s=size_limit
        if key==lg_na and s>fill_up and nlg_co<lg_co:
            if t==0:
                down+=1
            if nlg_co>=fill_up:
                group=group.sample(n=nlg_co,axis=0,ignore_index=True,random_state=42)
            else:
                group=group.sample(n=fill_up,axis=0,ignore_index=True,random_state=42)
        if s < fill_up:
            up+=1
            fill_up_list.append(key)
            N=int(fill_up/s)
            R=fill_up-N*s
            for n in range(N):
                li.append(group)
            if R > 0:
                if exact_fill_up:
                    li.append(group.sample(n=R,axis=0,ignore_index=True,random_state=42))
                else:
                    li.append(group)
        li.append(group)
    if down+up > 0:
        df=pd.concat(li,axis=0,ignore_index=True).sort_values(by=['KEY'],axis=0,ascending=True,ignore_index=True)
    if down > 0:
        print(f"{down}/{C} syscall groups have been down-sampled to group size {size_limit}")       
    else:
        print("No syscall groups have been down-sampled")
    if len(fill_up_list)>0:
        print(f"The following {len(fill_up_list)}/{C} syscalls have been upsampled to group size >= {fill_up}:")
        print(set([ int(s)  for s in fill_up_list ]))
        #print(set([ nr2sc[int(s)]  for s in fill_up_list ]))
    else:
        print("No syscall groups have been upsampled")
    print(f"Shape after group balancing: {df.shape}")
    Shape_Book(df.shape[0],df.shape[1],shape_count,"end group balancing")
    return df

# %%
def Validation_Image(precision,learning,validation,featimp_all,featimp,confusion):
    fig = plt.figure(figsize=(20, 30))
    fig.tight_layout()
    fig.suptitle('Validation Overview for '+experiment_name)
    # setting values to rows and column variables
    rows = 3
    columns = 2
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
    # showing image
    if precision!=None and not_missing_or_empty(precision):
        Image1 = mpimg.imread(precision)        
        plt.imshow(Image1)
        plt.axis('off')
        plt.title("Precision Recall")
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    # showing image
    if learning!=None and not_missing_or_empty(learning):
        Image2 = mpimg.imread(learning)
        plt.imshow(Image2)
        plt.axis('off')
        plt.title("Learning Curve")
    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)
    # showing image
    if validation!=None and not_missing_or_empty(validation):
        Image3 = mpimg.imread(validation)
        plt.imshow(Image3)
        plt.axis('off')
        plt.title("Validation Curve")
    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 4)
    # showing image
    if featimp_all!=None and not_missing_or_empty(featimp_all):
        Image4 = mpimg.imread(featimp_all)
        plt.imshow(Image4)
        plt.axis('off')
        plt.title("Feature Importance (All)")
    # Adds a subplot at the 5th position
    fig.add_subplot(rows, columns, 5)
    if featimp!=None and not_missing_or_empty(featimp):
        Image5 = mpimg.imread(featimp)      
       # showing image
        plt.imshow(Image5)
        plt.axis('off')
        plt.title("Feature Importance")
    # Adds a subplot at the 6th position
    fig.add_subplot(rows, columns, 6)
    # showing image
    if confusion!=None and not_missing_or_empty(confusion):
        Image6 = mpimg.imread(confusion)
        if version == '2.3.5':
            plt.imshow(Image6[14000:,14150:])
        else:
            plt.imshow(Image6)
        plt.axis('off')
        plt.title("Confusion Matrix")
        plt.savefig('Validation_Image.png')
    plt.close('all') 

# %%
def Shape_Book(nrows,ncolumns,shape_count,text):
    shape_count+=1
    df_shape = pd.DataFrame([[channel,nfeat,blur,nrows,ncolumns,shape_count,text]] ,columns=['CHANNEL', 'NFEAT', 'BLUR','N_ROW','N_COL','SHAPE_COUNT','COMMENT'])
    df_shape.to_csv(shape_na,mode='a') 


# %%
def Save_Config(yes,CNR2NR):
    if not yes:
        return
    X_train=pycaret.classification.get_config('X_train')
    y_train=pycaret.classification.get_config('y_train')
    X_test=pycaret.classification.get_config('X_test')
    y_test=pycaret.classification.get_config('y_test')
    if store_train_and_test_seperately:
        X_train.to_hdf(experiment_name+'.X_train.h5', key='OFF_SET',mode='w',complib='blosc:lz4',complevel=9)
        y_train.to_hdf(experiment_name+'.y_train.h5', key='LABEL',mode='w',complib='blosc:lz4',complevel=9)
        y_train_non_compact=y_train.replace(CNR2NR)
        y_train_non_compact.to_hdf(experiment_name+'.y_train_non_compact.h5', key='LABEL',mode='w',complib='blosc:lz4',complevel=9)
        X_test.to_hdf(experiment_name+'.X_test.h5', key='OFF_SET',mode='w',complib='blosc:lz4',complevel=9)
        y_test.to_hdf(experiment_name+'.y_test.h5', key='LABEL',mode='w',complib='blosc:lz4',complevel=9)
        y_test_non_compact=y_test.replace(CNR2NR)
        y_test_non_compact.to_hdf(experiment_name+'.y_train_non_compact.h5', key='LABEL',mode='w',complib='blosc:lz4',complevel=9)
    pycaret.classification.save_config(experiment_name+'.py_caret_setup_config.pkl')
    Shape_Book(X_train.shape[0]+X_test.shape[0],X_train.shape[1]+1,shape_count,"after pycaret preprocessing")
    return X_train,y_train,X_test,y_test

# %%
def Create_and_Save_CM(model,model_name,t):
    from sklearn.metrics import confusion_matrix
    print(f'Explicit creation of CM using {model_name} on set {t} ...')
    try:
        y_predicted=predict_model(model)
    except Exception as e:
        print(f"Prediction with {model_name} on set {t} failed: {e}")
        print(f'Explicit creation of CM using{model_name} on {t} failed')
    else:
        y_predicted.to_hdf(experiment_name+'.Y_PREDICTED.'+t+'.h5', key='Y_PREDICTED',mode='w',complib='blosc:lz4',complevel=9)
        cm=confusion_matrix(y_predicted['KEY'],y_predicted['Label'],normalize='true')
        print(cm)
        F=np.zeros(cm.shape,dtype=bool)
        T=np.ones(cm.shape,dtype=bool)
        plt.rcParams["figure.figsize"] = (30,30)
        cm_cmap='Blues'
        ax=sns.heatmap(cm,annot=False,cmap=cm_cmap)
        ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.plot()
        plt.savefig(experiment_name+'.CM.'+t+'.png')
        plt.close()
        pd.DataFrame(data=cm).to_hdf(experiment_name+'.CM.'+t+'.h5', key='CM',mode='w',complib='blosc:lz4',complevel=9)
        print(f'Explicit creation of CM using {model_name} on {t} succeeded')

# %%
def Eda_Plots(DATA,CNR2NR):
    #box_plot=False
    if box_plot:
        #box plot
        tic = time.perf_counter()
        print(f"Box plot ...")
        fig, out_fig = plt.subplots(figsize = (15,60))
        plt.xticks(rotation=45)
        plt.title("Box Plot of Train Matrix Features", fontsize=22)
        out_fig = sns.boxplot(data = DATA.drop('KEY', axis = 1), orient="h", fliersize=1, palette='crest')
        fig.savefig(experiment_name+'.BOX.png')
        plt.close()
        toc = time.perf_counter()
        print(f"Box plot took {toc - tic:0.4f} seconds")
    #violin_plot=False
    if violin_plot:
        #violin plot
        tic = time.perf_counter()
        print(f"Violin plot ...")
        fig, out_fig = plt.subplots(figsize = (15,100))
        plt.xticks(rotation=45)
        plt.title("Violoin Plot of Train Matrix Features", fontsize=22)
        out_fig = sns.violinplot(data = DATA.drop('KEY', axis = 1), orient="h", fliersize=1, palette="crest")
        fig.savefig(experiment_name+'.VIOLIN.png')
        plt.close()
        toc = time.perf_counter()
        print(f"Violin plot took {toc - tic:0.4f} seconds")
    if count_plot:
        # counting unique offsets per feature
        tic = time.perf_counter()
        print(f"Unique offset count per feature ...")
        df_all = pd.concat([DATA.drop('KEY', axis = 1), DATA], axis = 0)
        unique_df = pd.DataFrame(df_all.nunique()).reset_index()
        unique_df.columns=['features','count']
        fig, feat_bar = plt.subplots(figsize = (15,30))
        plt.title("Unique Offset Count per feature", fontsize=22)
        feat_bar = sns.barplot(y="features", x="count", data = unique_df, palette="crest", orient='h')
        fig.savefig(experiment_name+'.COUNT.png')
        plt.close('all')    
        toc = time.perf_counter()
        print(f"Unique offsetcount per feature took {toc - tic:0.4f} seconds")
    if profile_plot:
        #Distributions per label group
        tic = time.perf_counter()
        print(f"Plotting distinct offset distributions per label grouped per feature ...")
        C=DATA.shape[1]
        nc=5
        R=int((C-1)/5)+1
        fig, axes = plt.subplots(R,nc,figsize=(2*nc,2*(R-1)))
        fig.supxlabel('Distinct Offset Distributions per Label grouped per Feature', ha='center', fontweight='bold')
        target_order = sorted(DATA['KEY'].unique())
        for idx, ax in zip(range(1,C), axes.flatten()):
            cnt = DATA[f'MAD_{idx}'].value_counts().sort_index()
            sns.kdeplot(x=f'MAD_{idx}', 
                hue='KEY', 
                hue_order=target_order,
                data=DATA,
                alpha=0.5, 
                linewidth=0.1, 
                fill=True,
                thresh=0.05,
                legend=False,
                warn_singular=False,
                cumulative=True,
                log_scale=False,
                ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.spines['left'].set_visible(False)
            cnt = len(DATA[f'MAD_{idx}'].unique())
            ax.set_title(f'MAD_{idx}({cnt})', loc='right', weight='bold', fontsize=11)
        #delete empty subplots in last row
        d=0
        for i in range(nc*R,C-1,-1):
            d-=1
            axes.flatten()[d].axis('off')    
        fig.tight_layout()
        fig.savefig(experiment_name+'.PROFILES.png')
        plt.close()    
        toc = time.perf_counter()
        print(f"Plot of distinct offset distributions per label grouped per feature took {toc - tic:0.4f} seconds")
    if mean_profile_plot:
        #Mean offset profile per label group
        tic = time.perf_counter()
        print(f"Plotting mean offset profile per label grouped per feature ...")
        C=DATA.shape[1]
        fig, axes = plt.subplots(C, 1, figsize=(10, 2*C))
        target_order = sorted(DATA['KEY'].unique())
        mean = DATA.groupby('KEY').mean().sort_index()
        std = DATA.groupby('KEY').std().sort_index()
        for idx, ax in zip(range(1,C), axes.flatten()):
            x = np.arange(len(mean[f'MAD_{idx}'].index))
            ax.bar(x, mean[f'MAD_{idx}'],
                yerr = std[f'MAD_{idx}'], 
                width=2,edgecolor=None)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.margins(0.1)
            ax.spines['left'].set_visible(False)
            ax.set_title(f'MAD_{idx}', loc='right', weight='bold', fontsize=11)
        fig.supxlabel('Mean Offset Profile per Label grouped per feature', ha='center', fontweight='bold')
        fig.tight_layout()
        fig.savefig(experiment_name+'.MEAN_PROFILES.png')
        plt.close()    
        toc = time.perf_counter()
        print(f"Plot of mean offset profile per label group took {toc - tic:0.4f} seconds")
    if feature_correlation_plot:
        #feature correlation
        tic = time.perf_counter()
        print(f"Plotting feature correlation matrix ...")
        C=DATA.shape[1]
        fig, ax = plt.subplots(figsize=(0.2*C, 0.2*C))
        corr = DATA.drop('KEY', axis = 1).corr()
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr,
            square=True, center=0, linewidth=0.2,
            cmap=sns.diverging_palette(240, 10, as_cmap=True),
            mask=mask, ax=ax) 
        ax.set_title('Feature Correlation', loc='left', fontweight='bold')
        fig.savefig(experiment_name+'.FEATURE_CORRELATION_MATRIX.png')
        plt.close()    
        toc = time.perf_counter()
        print(f"Plot of feature correlation matrix took {toc - tic:0.4f} seconds")
    #dark color scheme
    from cycler import cycler   
    raw_light_palette = [
    (0, 122, 255), # Blue
    (255, 149, 0), # Orange
    (52, 199, 89), # Green
    (255, 59, 48), # Red
    (175, 82, 222),# Purple
    (255, 45, 85), # Pink
    (88, 86, 214), # Indigo
    (90, 200, 250),# Teal
    (255, 204, 0)  # Yellow
    ]
    raw_dark_palette = [
    (10, 132, 255), # Blue
    (255, 159, 10), # Orange
    (48, 209, 88),  # Green
    (255, 69, 58),  # Red
    (191, 90, 242), # Purple
    (94, 92, 230),  # Indigo
    (255, 55, 95),  # Pink
    (100, 210, 255),# Teal
    (255, 214, 10)  # Yellow
    ]
    raw_gray_light_palette = [
    (142, 142, 147),# Gray
    (174, 174, 178),# Gray (2)
    (199, 199, 204),# Gray (3)
    (209, 209, 214),# Gray (4)
    (229, 229, 234),# Gray (5)
    (242, 242, 247),# Gray (6)
    ]
    raw_gray_dark_palette = [
    (142, 142, 147),# Gray
    (99, 99, 102),  # Gray (2)
    (72, 72, 74),   # Gray (3)
    (58, 58, 60),   # Gray (4)
    (44, 44, 46),   # Gray (5)
    (28, 28, 39),   # Gray (6)
    ]
    light_palette = np.array(raw_light_palette)/255
    dark_palette = np.array(raw_dark_palette)/255
    gray_light_palette = np.array(raw_gray_light_palette)/255
    gray_dark_palette = np.array(raw_gray_dark_palette)/255
    white_color = gray_light_palette[-2]
    if set_dark_cs:
        mpl.rcParams['axes.prop_cycle'] = cycler('color',dark_palette)
        mpl.rcParams['figure.facecolor']  = gray_dark_palette[-2]
        mpl.rcParams['figure.edgecolor']  = gray_dark_palette[-2]
        mpl.rcParams['axes.facecolor'] =  gray_dark_palette[-2]
        mpl.rcParams['text.color'] = white_color
        mpl.rcParams['axes.labelcolor'] = white_color
        mpl.rcParams['axes.edgecolor'] = white_color
        mpl.rcParams['xtick.color'] = white_color
        mpl.rcParams['ytick.color'] = white_color
        mpl.rcParams['figure.dpi'] = 200
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.spines.right'] = False
    if umap_plot:
        from umap import UMAP
        #umap clustering
        sample_size=100000
        tic = time.perf_counter()
        print(f"UMAP clustering ...")
        if DATA.shape[0] <= sample_size:
            DATA_sub=DATA
        else:
            DATA_sub = DATA.sample(sample_size, random_state=0)
        target = DATA_sub['KEY'].astype('int')
        umap = UMAP(random_state=0)
        dr = umap.fit_transform(DATA_sub.iloc[:,:-1], target)
        dr_data = np.vstack((dr.T, DATA_sub.KEY)).T
        dr_df = pd.DataFrame(data=dr_data, columns=("D1", "D2", "KEY"))
        dr_df.to_hdf(experiment_name+'.UMAP_PLOT_DATA.h5', key='D1_D2_KEY',mode='w',complib='blosc:lz4',complevel=9)
        toc = time.perf_counter()
        print(f"UMAP clustering took {toc - tic:0.4f} seconds")
        #Plotting
        tic = time.perf_counter()
        print(f"Plotting 2D UMAP clusters ...")
        W=12
        nj=10
        Size=1
        L=len(DATA_sub['KEY'].unique())
        add_rows=int((L-1)/nj+1)
        ni=nj+add_rows
        H=W+(W-1)*int(add_rows/nj)
        fig = plt.figure(figsize=(W,H*1.2))
        gs = fig.add_gridspec(ni, nj)
        ax = fig.add_subplot(gs[:-add_rows,:])
        sub_axes=[None]*ni*nj
        idx=-1
        for i in range(-add_rows,0):
            for j in range(nj):
                idx=idx+1
                if(idx==L):
                    break
                sub_axes[idx] = fig.add_subplot(gs[i,j])
        for idx in range(L):
            ax.scatter(x=dr[:,0][target==idx], y=dr[:,1][target==idx],s=Size, alpha=0.2)
            for j in range(L):
                sub_axes[j].scatter(x=dr[:,0][target==idx], y=dr[:,1][target==idx],
                            s=Size if idx==j else 0.1, 
                            alpha = 0.4 if idx==j else 0.008, 
                            color = (dark_palette[j%9]) if idx==j else white_color,
                            zorder=(idx==j)
                           )   
            sub_axes[idx].set_xticks([])
            sub_axes[idx].set_yticks([])
            sub_axes[idx].set_xlabel('')
            sub_axes[idx].set_ylabel('')
            #sub_axes[idx].set_title(f'SC_{idx+1}')
            scna=nr2sc[CNR2NR[idx]][0:12]
            print(f'{idx+1}: {scna} plotted')
            sub_axes[idx].set_title(f'{scna}')
            sub_axes[idx].spines['right'].set_visible(True)
            sub_axes[idx].spines['top'].set_visible(True)  
        ax.set_title('UMAP Offset Distribution (2D) ', fontweight='bold', fontfamily='serif', fontsize=20, loc='left')       
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        fig.tight_layout()
        fig.savefig(experiment_name+'.UMAP.png')
        plt.close()    
        toc = time.perf_counter()
        print(f"UMAP 2D cluster plot took {toc - tic:0.4f} seconds")

# %%
def read_multiblock_data(sna,channel,nfeat,blur):
        base_na=channel+"_"+str(nfeat)+"_"+str(blur)
        tna="./"+base_na+'_0.oneblock.h5'
        T=tna.split('_')
        with h5py.File(sna,'r') as hdf:
            ls=list(hdf.keys())
            non_scs=False
            if not_missing_or_empty(tna):
                os.remove(tna)
                print(f"Existing{tna} removed")
            G=0
            for g in ls:
                s=g.split('_')
                if(len(s)!=10):
                    continue
                G+=1
                if s[1]==str(65535):
                    non_scs=True
                df=pd.read_hdf(sna,key=g)
                if df.shape[1]<nfeat:
                    print(f'Warning: number of features in {sna} = {df.shape[1]} < {nfeat}')
                    print(f'Warning: {sna} cannot be used for {channel}_{nefeat}_{blur}')
                    df=pd.DataFrame()
                    return df
                else:
                    df=df.iloc[:,0:nfeat]
                df.to_hdf(tna,key='OFF_SET',mode='a',format='table',append=True,complib='blosc:lz4',complevel=9)
                #print(f"syscall group <{g}> added to {tna}")
            if not non_scs:
                print(f'Warning: non-syscall group missing in {sna}')
            df=pd.read_hdf(tna)
            os.remove(tna)
            return df

# %%
def run_group_training(na,fna,experiment_name,size_cut,n_trials,use_existing_feature_rejections,load_model_path):
    tic = time.perf_counter()
    print(f"Reading {fna} ...")
    try:
        DATA=pd.read_hdf(fna)
    except Exception as e:
        if str(e).find('key must be provided when HDF5 file contains multiple datasets') ==-1:
            print(e)
        else:
            DATA=read_multiblock_data(fna,channel,nfeat,blur)
    if DATA.empty:
        print(f'Warning: could not enter any data from {fna}')
        return
    toc = time.perf_counter()
    print(f"Reading {fna} took {toc - tic:0.4f} seconds")
    #add key in read-in data column to eval_dict
    #
    eval_dict_columns.append('H5_Input')
    KEY_LIST=DATA['KEY'].unique()
    KEY_LIST_not_covered=set(KEY_LIST)
    for key in eval_dict.keys():
        if eval_dict[key][0] in KEY_LIST:
            KEY_LIST_not_covered.remove(eval_dict[key][0])
            eval_dict[key].append(True)
        else:
            eval_dict[key].append(False)
    if KEY_LIST_not_covered != set():
        print('Warning: The following syscall indices occuring in input data are not covered by basic key list:')
        print(KEY_LIST_not_covered)
    #
    print("")
    tic = time.perf_counter()
    print("Balancing KEY groups ...")
    DATA = Balance_Groups(DATA,fill_up)
    toc = time.perf_counter()
    print(f"Balancing KEY groups took {toc - tic:0.4f} seconds")
    size_profile=Size_Profile(DATA)
    size_profile.to_csv(experiment_name+'.size_profile.csv',index=False)
    print("Size profile:")
    print(size_profile)
    print("")
    print("The 10 largest KEY groups are:")
    NL=size_profile.nlargest(10,'COUNT')
    print(NL)
    if train_test_split:
        print("")
        tic=time.perf_counter()
        print("Splitting into Train / Test data ...")
        y=DATA['KEY'].array
        X_train,X_test,y_train,y_test=train_test_split(DATA.iloc[:,1:],y,stratify=y,test_size=0.25,random_state=42)
        y_train=pd.DataFrame(data=y_train,columns=['KEY'])
        X_train=X_train.reset_index(drop=True)
        DATA=pd.concat([y_train,X_train],axis=1)
        del(y_train)
        del(X_train)
        y_test=pd.DataFrame(data=y_test,columns=['KEY'])
        X_test=X_test.reset_index(drop=True)
        TEST=pd.concat([y_test,X_test],axis=1)
        TEST.sort_values(by=['KEY'],ascending=True,inplace=True)
        TEST.to_hdf(experiment_name+'.test.h5', key='OFF_SET',mode='w',complib='blosc:lz4',complevel=9)
        del(y_test)
        del(X_test)
        del(TEST)
        toc = time.perf_counter()
        print(f"Splitting took {toc - tic:0.4f} seconds")
    if intergroup_drop!="":
        print("")
        #drop inter-group duplicates from DATA
        feat_list=DATA.columns.tolist()
        feat_list.pop(0)
        print(f"Shape of train data before drop of inter-group duplicates {DATA.shape}")
        DATA=DATA.drop_duplicates(subset=feat_list,ignore_index=True,keep=intergroup_drop)
        DATA=DATA.sort_values(by=['KEY'],axis=0,ascending=True,ignore_index=True)
        DATA.sort_values(by=['KEY'],ascending=True,inplace=True)
        print(f"Shape of train data after  drop of inter-group duplicates {DATA.shape}")
        Shape_Book(DATA.shape[0],DATA.shape[1],shape_count,"after inter-group drop of key duplicates")
        print("")
    else:
        print("")
        print("Drop of inter-group duplicates has been switched off")
        print("")
    #
    DATA['KEY'] = DATA['KEY'].apply(lambda x : -x if x < 0 else x)
    #
    tic = time.perf_counter()
    print("Downcasting numeric types ...")
    fcols = DATA.select_dtypes('float').columns
    icols = DATA.select_dtypes('integer').columns
    DATA[fcols] = DATA[fcols].apply(pd.to_numeric, downcast='float')
    DATA[icols] = DATA[icols].apply(pd.to_numeric, downcast='integer')
    toc = time.perf_counter()
    print(f"Downcasting numeric types took {toc - tic:0.4f} seconds")
    #
    print(DATA.info())
    #
    print(LS_paper,'train labels in the training data of the paper')  
    LS=label_set(DATA,'in the current training data after duplicate drop')
    if 65535 not in LS and 32768 not in LS:
        if stop_at_missing_nonsc:
            raise SystemExit('Error: No non-syscalls found.')
        else:
         print('Warning: no non-syscalls occur in dataset')
    if len(LS)<LS_paper:
        print('Warning: current training data contain smaller label set than in the paper')
    Start_Label_Set=set(natsorted([nr2sc.get(i) for i in LS ]))
    print("")
    print("syscall set at start:")
    print(Start_Label_Set)
    print('The paper assumes that the following labels are missing in the available train data:')
    print(natsorted(missing_nr))
    compare_label_sets(missing_nr,LS,'missing','available train')
    print("")
    print('The paper assumes that the following labels are excluded from the available train data:')
    print(natsorted(excluded_nr))
    compare_label_sets(excluded_nr,LS,'excluded','available train')
    #
    print(size_cut)
    if size_cut=='paper':
        if(len(excluded)>0):
            DATA_S=DATA.loc[DATA['KEY'].isin(excluded) ]
            LS_S=label_set(DATA_S,'exluded by paper') 
            #DATA_S.to_csv(experiment_name+'.DATA_S.csv.zip',index=0)
            DATA_L=DATA.loc[~ DATA['KEY'].isin(excluded) ] 
        else:
            os.system('rm -f DATA_S.csv.zip')
            DATA_L=DATA
            #DATA_L.to_csv(experiment_name+'.DATA_L.csv.zip',index=0)
        LS_L=label_set(DATA_L,'not exluded by paper')
    else:
        if size_cut=='auto':
            _,size_cut=DATA.shape
            size_cut=size_cut-1
            print('size cut is set to',size_cut)
            if size_cut_upper_limit>0 and size_cut>size_cut_upper_limit:
                size_cut=size_cut_upper_limit
                print('size cut is limited to',size_cut_upper_limit,'in auto mode')
        if size_cut > 0:
            print('size cut is set to',size_cut)
            DATA_S=DATA.groupby('KEY').filter(lambda x: len(x) < size_cut)
            #DATA_S.to_csv(experiment_name+'.DATA_S.csv.zip',index=0)
            LS_S=label_set(DATA_S,'below size '+str(size_cut))
            DATA_L=DATA.groupby('KEY').filter(lambda x: len(x) >= size_cut)
        else:
            os.system('rm -f DATA_S.csv.zip')
            DATA_L=DATA
        #DATA_L.to_csv(experiment_name+'.DATA_L.csv.zip',index=0)
        LS_L=label_set(DATA_L,'above size '+str(size_cut))
    #downcasting DATA_L
    fcols = DATA_L.select_dtypes('float').columns
    icols = DATA_L.select_dtypes('integer').columns
    DATA_L[fcols] = DATA_L[fcols].apply(pd.to_numeric, downcast='float')
    DATA_L[icols] = DATA_L[icols].apply(pd.to_numeric, downcast='integer')
    #
    # minimum_paper_size_cut=smallest sample_size of all syscalls in explicitly_mentioned_sc
    mask=(size_profile['NAME'].isin(explicitly_mentioned_sc))
    size_profile_expl_ment=size_profile[mask]
    print(size_profile_expl_ment.shape)
    size_profile_expl_ment.sort_values(by=['COUNT'],ascending=True,inplace=False)
    print("Size profile of SCs explicitly shown in paper pictures:")
    print(size_profile_expl_ment.T)  
    min_consist_size_cut=size_profile_expl_ment['COUNT'].min()
    if size_cut != 'paper':
        if size_profile_expl_ment['COUNT'].min() < int(size_cut):
            print("Warning: maximum size cut allowed to cover all SCs in paper pictures:",min_consist_size_cut)
            print('Warning: The following SCs explicitly shown in paper pictures are not covered by current size cut of',str(size_cut)+':')
            not_covered=size_profile_expl_ment.mask(size_profile_expl_ment['COUNT'] >= size_cut)
            not_covered.dropna(inplace=True)
            not_covered=not_covered.astype({'ID': 'int32','COUNT': 'int32' })
            print(not_covered)
        else:
            print("OK: maximum size cut covers all SCs in paper pictures")
    plt.rcParams["figure.figsize"] = (15, 70)
    if size_cut=='paper' and len(excluded)>0:
        DATA_S["KEY"].replace(nr2sc).value_counts().plot.barh(legend=None,title="Syscalls with < "+str(size_cut)+" rows")
        plt.plot()
        plt.savefig(experiment_name+'.small_size_profile.png')
        plt.close()
    else:
        if size_cut > 0:
            DATA_S["KEY"].replace(nr2sc).value_counts().plot.barh(legend=None,title="Syscalls with < "+str(size_cut)+" rows")
            plt.plot()
            plt.savefig(experiment_name+'.small_size_profile.png')
            plt.close()
    DATA_L["KEY"].replace(nr2sc).value_counts().plot.barh(legend=None,title="Syscalls with >= "+str(size_cut)+" rows")
    plt.plot()
    plt.savefig(experiment_name+'.large_size_profile.png')
    plt.close()
    LS=label_set(DATA_L,'after drop of labels with row count <'+str(size_cut))
    compare_label_sets(missing_nr,LS,'missing','available train')
    print("")
    print('The paper assumes that the following labels are excluded from the available train data:')
    print(natsorted(excluded_nr))
    compare_label_sets(excluded_nr,LS,'excluded','available train')
    del(DATA) 
    gc.collect()
    class_weights=calc_weights(DATA_L,"after size cut")
    LSW=natsorted([l for l in class_weights.keys()])
    if LSW != LS:
        print("")
        print('WARNING: LSW and LS differ')
        print('LSW:')
        print(LSW)
        print('LS:')
        print(LS)
        compare_label_sets(LSW,LS,'weight','train')
        print("")
    
    # Tight packing of syscall NR due to cuml GPU requirements for some classifiers 
    #replace syscall keys by compact index list "CNR" starting from 0
    NR2CNR=dict(zip(LS,[n for n in range(0,len(LS))]))
    #add compact index to eval_dict
    eval_dict_columns.append('CNR')
    for key in eval_dict.keys():
        eval_dict[key].append(NR2CNR.get(eval_dict[key][0]))
    with open(experiment_name+'.NR2CNR.json', 'w') as file:
        json.dump(dict(zip( [str(n) for n in LS], [n for n in range(0,len(LS))])), file)
    print("NR -> CNR")
    print(NR2CNR)
    CNR2NR=dict(zip([n for n in range(0,len(LS))],LS))
    with open(experiment_name+'.CNR2NR.json', 'w') as file:
        json.dump(dict(zip([n for n in range(0,len(LS))],[str(n) for n in LS])), file)
    print("CNR -> NR")
    print(CNR2NR)    
    #
    #Final transformations required before training
    #Labels transformed to [0,...,#labels-1] (required by use_gpu=True, but generally applied)
    if log2_transform:
        tic = time.perf_counter()
        print(f"Log2 transform of offset data ...")
        LOGX=pd.DataFrame(logb(DATA_L.iloc[:,1:].to_numpy(copy=True)),columns=DATA_L.iloc[:,1:].columns)
        DATA_L.iloc[:,1:]=LOGX.apply(pd.to_numeric,downcast='float')
        del(LOGX) 
        gc.collect()
        print('log2 transform applied to offset data')
        toc = time.perf_counter()
        print(f"Log2 transform of offset data took {toc - tic:0.4f} seconds")
    #
    if eda_plots:
        Eda_Plots(DATA_L,CNR2NR)
    #
    #Final adjustments
    DATA_L["KEY"].replace(to_replace=NR2CNR,inplace=True)
    DATA_L.sort_values(by=['KEY'],inplace=True)
    fcols = DATA_L.select_dtypes('float').columns
    icols = DATA_L.select_dtypes('integer').columns
    if use_gpu == True:
        #transform int features to float (required for GPU processing)
        DATA_L[icols]=DATA_L[icols].apply(pd.to_numeric,downcast='float')
        DATA_L['KEY']=DATA_L['KEY'].astype('int')
        labels=DATA_L['KEY'].unique()
        print(labels)
    DATA_L[fcols] = DATA_L[fcols].apply(pd.to_numeric, downcast='float')
    DATA_L[icols] = DATA_L[icols].apply(pd.to_numeric, downcast='integer')
    if 'KEY' not in icols:
        DATA_L['KEY']=DATA_L['KEY'].astype('int')
        DATA_L['KEY']=DATA_L['KEY'].apply(pd.to_numeric,downcast='integer')
    #
    print("Data structure immediatly before training:")
    skim(DATA_L)
    #
    if load_model_path != "":
        print(f'Warning: model training is skipped due to existing model path {load_model_path}')
        return load_model_path
    #
    # Pycaret Baseline
    #
    tic = time.perf_counter()
    model_name='RandomForrestClassifier'
    print(f"Setting up pycaret model for {model_name} ...")
    experiment_name=experiment_name
    n_jobs_default=int(config_dict['n_jobs'])
    train_rf_baseline=setup(data=DATA_L,
                    target=target, 
                    session_id=session_id, 
                    verbose=verbose, 
                    normalize=normalize,
                    normalize_method=normalize_method,
                    fix_imbalance=fix_imbalance,
                    fix_imbalance_method=fix_imbalance_method,
                    feature_selection=feature_selection,
                    feature_selection_method=feature_selection_method,
                    ignore_low_variance=ignore_low_variance, 
                    remove_multicollinearity=remove_multicollinearity,
                    remove_outliers=remove_outliers,
                    create_clusters=create_clusters,
                    n_jobs=7,
                    use_gpu=use_gpu,
                    silent=True,
                    log_experiment=False,
                    experiment_name=experiment_name,
                    log_plots=log_plots,
                    log_profile=False,
                    log_data=False,
                    data_split_stratify=data_split_stratify,
                    fold_strategy=fold_strategy,
                    fold=fold
                   )
    toc = time.perf_counter()
    print(f"Pycaret model setup for {model_name} took {toc - tic:0.4f} seconds")
    #
    del(DATA_L) 
    gc.collect()
    #
    Save_Config(True,CNR2NR)
    #
    eval_dict_columns.append('In_Train')
    y_list=get_config('y_train').unique()
    for key in eval_dict.keys():
        if eval_dict[key][5] in y_list:
            eval_dict[key].append(True)
        else:
            eval_dict[key].append(False)
    #
    tic = time.perf_counter()
    print(f"Creating model for {model_name} ...")
    pycaret.classification.set_config('n_jobs_param',3)
    class_weight_method='balanced_subsample'
    #class_weight_method='dictionary'
    #
    class_weights=calc_weights(pd.concat([get_config('y_train'),get_config('X_train')],axis=1),"for train set")
    print(class_weights)
    #add weights of training data column to eval_dict
    eval_dict_columns.append('Train_Weight')
    for key in eval_dict.keys():
            eval_dict[key].append(class_weights.get(eval_dict[key][5]))
    #
    if use_gpu==False:
        if class_weight_method=='balanced' or class_weight_method=='balanced_subsample':
            rf = create_model('rf',class_weight=class_weight_method)
        elif class_weight_method=='dictionary':
            #calc_weights(DATA_TRAIN,"after pycaret preprocessing")
            rf = create_model('rf',class_weight=class_weights)
        else:
            raise SystemExit('Error: unknown class weight method')            
    else:
        print('Warning: cannot use weights due to cuML restrictions')
        rf = create_model('rf')
        raise SystemExit('Error: will not continue')
        #
    results=pycaret.classification.pull()
    results.to_csv(experiment_name+'.results.csv',index=0)
    save_model(rf,experiment_name+".model")
    toc = time.perf_counter()
    print(f"Model creation for {model_name} took {toc - tic:0.4f} seconds")
    #
    file_path='./'+experiment_name+'.model.txt'
    with open(file_path, "w") as o:
        with contextlib.redirect_stdout(o):
            print(rf)
    #
    Create_and_Save_CM(rf,model_name,'HOLD_OUT.BASE')
    #
    plot_failure=[]
    img_dict={}
    if plot_baseline:
        model_name='RF_BASELINE'
        print('------------------------------------------------------------------')
        for key in plot_dict:
            if not plot_dict[key][3]:
                continue
            if key == 'learning':
                pycaret.classification.set_config('n_jobs_param',5)
            if key == 'vc':
                pycaret.classification.set_config('n_jobs_param',5)
            sna="./'"+plot_dict[key][0]+plot_dict[key][2]+"'"
            tna="./"+experiment_name+'.'+plot_dict[key][1]+plot_dict[key][2]
            img_dict[key]=tna
            print(sna+" -> "+tna+" ...")
            try:
                tic = time.perf_counter()
                print(f"Plotting <{plot_dict[key][0]}> for {model_name} ...")
                plot_model(rf,plot=key,save=True)
            except Exception as e:
                print(f"Plotting <{plot_dict[key][0]}> for {model_name} caused the following error: {e}")
                plot_failure.append([channel,nfeat,blur,plot_dict[key][0],e])
                if os.system('ls '+sna+' 2>/dev/null 1>/dev/null') == 0:
                    os.system('mv '+sna+' '+tna)
                    print(f"{sna} could be saved despite error: {e}")
                else:
                    print(f"{sna} not found")
            else:
                if os.system('ls '+sna+' 2>/dev/null') == 0:
                    os.system('mv '+sna+' '+tna)
                    print(f"{sna} moved to {tna}")
                else:
                    print(f"{sna} not found even, though it should exist")
                    plot_failure.append([channel,nfeat,blur,plot_dict[key][0],sna+" not found, even though it should exist"])
            pycaret.classification.set_config('n_jobs_param',int(n_jobs_default))
            print('------------------------------------------------------------------')
            print("")
        #
        Validation_Image(img_dict.get('pr'),img_dict.get('learning'),img_dict.get('vc'),img_dict.get('feature'),img_dict.get('feature_all'),img_dict.get('confusion_matrix'))
        if os.system('ls Validation_Image.png 2>/dev/null 1>/dev/null') == 0:
            os.system('mv Validation_Image.png '+"./"+experiment_name+'.val_summary.png')
        else:
            print("Validation_Image not found even though it should exist")
            plot_failure.append([channel,nfeat,blur,plot_dict[key][0],"Validation_Image not found even though it should exist"])
        if len(plot_failure)>0:
            df_plot_failure = pd.DataFrame(plot_failure,columns=['CHANNEL', 'NFEAT', 'BLUR','PLOT_NAME','ERROR_MESSAGE'])
            df_plot_failure.to_csv(plot_failure_na,mode='a')
    #
    pycaret.classification.set_config('n_jobs_param',int(n_jobs_default))
    #
    model='rf'
    if tune_baseline:
        tune_failure=[]
        pycaret.classification.set_config('n_jobs_param',4)
        for key,li in tune_dict.items():
            if not li[5]:
                continue
            tic = time.perf_counter()
            print(f"Tuning {model_name} with {key} options ...")
            tune_model_na=model_name+'_tuned_'+key
            tune_mod='rf_'+key
            try:
                print(f"Tuning {model_name} with optimize={li[1]},search_library={li[2]},search_algorithm={li[3]},early_stopping={li[4]} ...")
                if key=='default':
                    rf_default=tune_model(rf,n_iter=int(li[0]),optimize=li[1],search_library=li[2],search_algorithm=li[3],early_stopping=li[4],fit_kwargs={'class_weight':class_weights})
                    print(rf_default)
                    results_tuned=pycaret.classification.pull()
                    results_tuned.to_csv(experiment_name+'.results.'+key+'.tuned.csv',index=0)
                    file_path='./'+experiment_name+'.'+key+'.tuned.model.txt'
                    with open(file_path, "w") as o:
                        with contextlib.redirect_stdout(o):
                            print(rf_default)
                    save_model(rf_default,experiment_name+'.'+key+'.tuned.model')
                elif key=='scikit_optimize':
                    rf_scikit_optimize=tune_model(rf,n_iter=int(li[0]),optimize=li[1],search_library=li[2],search_algorithm=li[3],early_stopping=li[4])
                    print(rf_scikit_optimize)
                    results_tuned=pycaret.classification.pull()
                    results_tuned.to_csv(experiment_name+'.results.'+key+'.tuned.csv',index=0)
                    file_path='./'+experiment_name+'.'+key+'.tuned.model.txt'
                    with open(file_path, "w") as o:
                        with contextlib.redirect_stdout(o):
                            print(rf_scikit_optimize)
                    save_model(rf_scikit_optimize,experiment_name+'.'+key+'.tuned.model')
                elif key=='sklearn_bayesian':
                    rf_sklearn_bayesian=tune_model(rf,n_iter=int(li[0]),optimize=li[1],search_library=li[2],search_algorithm=li[3],early_stopping=li[4])
                    print(rf_sklearn_bayesian)
                    results_tuned=pycaret.classification.pull()
                    results_tuned.to_csv(experiment_name+'.results.'+key+'.tuned.csv',index=0)
                    file_path='./'+experiment_name+'.'+key+'.tuned.model.txt'
                    with open(file_path, "w") as o:
                        with contextlib.redirect_stdout(o):
                            print(rf_sklearn_bayesian)
                    save_model(rf_sklearn_bayesian,experiment_name+'.'+key+'.tuned.model')
                elif key=='sklearn_hyperopt':
                    rf_sklearn_hyperopt=tune_model(rf,n_iter=int(li[0]),optimize=li[1],search_library=li[2],search_algorithm=li[3],early_stopping=li[4])
                    print(rf_sklearn_hyperopt)
                    results_tuned=pycaret.classification.pull()
                    results_tuned.to_csv(experiment_name+'.results.'+key+'.tuned.csv',index=0)
                    file_path='./'+experiment_name+'.'+key+'.tuned.model.txt'
                    with open(file_path, "w") as o:
                        with contextlib.redirect_stdout(o):
                            print(rf_sklearn_hyperopt)
                    save_model(rf_sklearn_hyperopt,experiment_name+'.'+key+'.tuned.model')
                elif key=='sklearn_optuna':
                    rf_sklearn_optuna=tune_model(rf,n_iter=int(li[0]),optimize=li[1],search_library=li[2],search_algorithm=li[3],early_stopping=li[4])
                    print(rf_sklearn_optuna)
                    results_tuned=pycaret.classification.pull()
                    results_tuned.to_csv(experiment_name+'.results.'+key+'.tuned.csv',index=0)
                    file_path='./'+experiment_name+'.'+key+'.tuned.model.txt'
                    with open(file_path, "w") as o:
                        with contextlib.redirect_stdout(o):
                            print(rf_sklearn_optuna)
                    save_model(rf_sklearn_optuna,experiment_name+'.'+key+'.tuned.model')
                elif key=='optuna':
                    rf_optuna=tune_model(rf,n_iter=int(li[0]),optimize=li[1],search_library=li[2],search_algorithm=li[3],early_stopping=li[4])
                    print(rf_optuna)
                    results_tuned=pycaret.classification.pull()
                    results_tuned.to_csv(experiment_name+'.results.'+key+'.tuned.csv',index=0)
                    file_path='./'+experiment_name+'.'+key+'.tuned.model.txt'
                    with open(file_path, "w") as o:
                        with contextlib.redirect_stdout(o):
                            print(rf_optuna)
                    save_model(rf_optuna,experiment_name+'.'+key+'.tuned.model')
                else:
                    e=f"Unknown tuning model {key}"
                    tune_failure.append([channel,nfeat,blur,li[1],li[2],li[3],e])
                    print(f"Error: Tuning {model_name} with optimize={li[1]},search_library={li[2]},search_algorithm={li[3]}] failed: {e}")
                    #exec(tune_mod+"=tune_model("+str(model)+",n_iter="+str(li[0])+",optimize="+li[1]+",search_library="+li[2]+",search_algorithm="+li[3]+",early_stopping="+li[4]+")")
            except Exception as e:
                tune_failure.append([channel,nfeat,blur,li[1],li[2],li[3],e])
                print(f"Error: Tuning {model_name} with optimize={li[1]},search_library={li[2]},search_algorithm={li[3]}] failed: {e}")
            toc = time.perf_counter()
            print(f"Tuning {model_name} with optimize={li[1]},search_library={li[2]},search_algorithm={li[3]}],early_stopping={li[4]} options took {toc - tic:0.4f} seconds")
            time.sleep(5)
        #
        if len(tune_failure)>0:
            df_tune_failure = pd.DataFrame(tune_failure,columns=['CHANNEL', 'NFEAT', 'BLUR','OPT','LIB','SEARCH','ERROR_MESSAGE'])
            df_tune_failure.to_csv(tune_failure_na,mode='a')
    #
    pycaret.classification.set_config('n_jobs_param',int(n_jobs_default))
    #
    tic = time.perf_counter()
    print(f"Creating the best model for {Metric} ...")
    best=pycaret.classification.automl(optimize='Accuracy')
    print(best)
    best_na='BEST_ACCURACY'
    file_path='./'+experiment_name+'.'+best_na+'.model.txt'
    with open(file_path, "w") as o:
        with contextlib.redirect_stdout(o):
            print(best)
    save_model(best,experiment_name+'.'+best_na+'.model')
    toc = time.perf_counter()
    print(f"Creation of the best model for {Metric} took {toc - tic:0.4f} seconds")
    pycaret.classification.set_config('n_jobs_param',2)
    tic = time.perf_counter()
    print("Finalizing the best model ...")
    final_rf = finalize_model(best)
    save_model(final_rf,experiment_name+".final.model")
    toc = time.perf_counter()
    print(f"Finalization of the best model took {toc - tic:0.4f} seconds")
    return experiment_name+".final.model.pkl"

# %%
def final_evaluation(tana,eval_dict_columns,eval_dict,testdata_path):
    if not not_missing_or_empty(tana):
        print(f'Warning: the input file for final evaluation {tana} is empty or missing')
        print('Warning: no final evaluation is performed')
        return       
    for key in eval_dict.keys():
        eval_dict[key]=eval_dict[key][0:7]
    eval_dict_columns=eval_dict_columns[0:7]
    tic=time.perf_counter()
    print(f'Reading {tana} ...')        
    df=pd.read_hdf(tana)
    #add <occurs in test data column> to eval_dict
    key_list=df['KEY'].unique()
    key_list_not_covered=set(key_list)
    overwrite=""
    colna='In_Test'
    if colna not in eval_dict_columns:
        eval_dict_columns.append(colna)
    else:
        overwrite=eval_dict_columns.index(colna)
    for key in eval_dict.keys():
        if eval_dict[key][0] in key_list:
            key_list_not_covered.remove(eval_dict[key][0])
            if overwrite=="":
                eval_dict[key].append(True)
            else:
                eval_dict[key][overwrite]=True 
        else:
            if overwrite=="":
                eval_dict[key].append(False)
            else:
                eval_dict[key][overwrite]=False                
    if key_list_not_covered != set():
        print('Warning: The following syscall indices occuring in the test data are not covered by basic key list:')
        print(key_list_not_covered)
    toc = time.perf_counter()
    print(f'Reading {tana} took {toc - tic:0.4f} seconds')
    #
    #add <True Positive> column to eval_dict
    tic = time.perf_counter()
    print('Start calculating syscall metrics ....')
    print("------------------------------------------------------------")
    print("Counting True Positives ...")
    TP=df['KEY'].value_counts()
    key_list=df['KEY'].unique()
    key_list_not_covered=set(key_list)
    owi=""
    colna='TP'
    if colna not in eval_dict_columns:
        eval_dict_columns.append(colna)
    else:
        owi=eval_dict_columns.index(colna)
    for key in eval_dict.keys():
        i=eval_dict[key][0]
        if i in key_list:
            key_list_not_covered.remove(i)
            if owi != "":
                eval_dict[key][owi]=int(TP[i])
                key_list_not_covered.remove(i)
            else:
                eval_dict[key].append(TP[i])
    if key_list_not_covered != set():
            for i in list(key_list_not_covered):
                print(f'Warning: TP count {TP[i]} for syscall index {i} not added to evaluation dictionary')
                print(f'Warning: {i} does not occur as index in the basic syscall list')
    #
    #add <True Negative> column to eval_dict
    print("------------------------------------------------------------")
    print("Counting True Negatives ...")
    TN=-TP
    TN=TN.add(len(df['KEY']))
    key_list_not_covered=set(key_list)
    owi=""
    colna='TN'
    if colna not in eval_dict_columns:
        eval_dict_columns.append(colna)
    else:
        owi=eval_dict_columns.index(colna)
    for key in eval_dict.keys():
        i=eval_dict[key][0]
        if i in key_list:
            key_list_not_covered.remove(i)
            if owi != "":
                eval_dict[key][owi]=TN[i]
            else:
                eval_dict[key].append(TN[i])
    if key_list_not_covered != set():
        for i in list(key_list_not_covered):
            print(f'Warning: TN count {TN[i]} for syscall index {i} not added to evaluation dictionary')
            print(f'Warning: {i} does not occur as index in the basic syscall list')
    #
    #add <False Negative> column to eval_dict
    print("------------------------------------------------------------")
    print("Counting False Negatives ...")
    fig=plt.figure(figsize=(20, 20))
    FN_dir='./FN_dir/'
    pathlib.Path(FN_dir).mkdir(parents=True, exist_ok=True)
    FN={}
    total_fn_miss={}
    fn_class_count={}
    key_grouped=df.groupby('KEY')
    for key,group in key_grouped:
        total_fn_miss[key]=None
        fn_class_count[key]=0
        kc=group['Label'].value_counts()
        tp=kc.get(key)
        if tp == None:
            print(f'Warning: {key} has no true positives')
            FN[key]=group.shape[0]
        else:
            FN[key]=group.shape[0]-tp
        if kc.count == 1 and kc[0]!=key:
            print(f'Warning: {key} is completely mapped to false negative class {kc[0]}')
            total_fn_miss[key]=kc[0]
            fn_class_count[key]=1
        else:
            if kc.count()>1:
                fn_class_count[key]=kc.count()-1
                print(f'{key} has {fn_class_count[key]} false negative classes')
                group["Label"].value_counts().plot.bar(legend=None,title=f"False negatives {key}")
                plt.plot()
                plt.savefig(FN_dir+experiment_name+'.'+str(key)+'.FN_SPREAD.png')
                plt.close('all')
    for colna in ['FN', 'total_fn_miss', 'fn_class_count']:
        key_list_not_covered=set(key_list)
        owi=""
        if colna not in eval_dict_columns:
            eval_dict_columns.append(colna)
        else:
            owi=eval_dict_columns.index(colna)
        for key in eval_dict.keys():
            i=eval_dict[key][0]
            if i in key_list:
                bv=eval(colna+'['+str(i)+']')
                key_list_not_covered.remove(i)
                if owi != "":
                    eval_dict[key][owi]=bv
                else:
                    eval_dict[key].append(bv)
        if key_list_not_covered != set():
            for i in list(key_list_not_covered):
                bv=eval(colna+'['+str(i)+']')
                print(f'Warning: {colna} count {bv} for syscall index {i} not added to evaluation dictionary')
                print(f'Warning: {i} does not occur as index in the basic syscall list')
    #
    #add <False Positive> column to eval_dict
    print("------------------------------------------------------------")
    print("Counting False Positives ...")
    fig=plt.figure(figsize=(20, 20))
    FP_dir='./FP_dir/'
    pathlib.Path(FP_dir).mkdir(parents=True, exist_ok=True)
    FP={}
    total_fp_miss={}
    fp_class_count={}
    label_grouped=df.groupby('Label')
    for label,group in label_grouped:
        total_fp_miss[label]=None
        fp_class_count[label]=0
        lc=group['KEY'].value_counts()
        tp=lc.get(label)
        if tp == None:
            print(f'Warning: {label} has no true positives')
            FP[label]=group.shape[0]
        else:
            FP[label]=group.shape[0]-tp
        if lc.count == 1 and lc[0]!=key:
            print(f'Warning: {label} is completely mapped to false positive class {lc[0]}')
            total_fp_miss[label]=lc[0]
            fp_class_count[label]=1
        else:
            if lc.count()>1:
                fp_class_count[label]=lc.count()-1
                print(f'{label} has {fp_class_count[label]} false positive classes')
                group["KEY"].value_counts().plot.bar(legend=None,title=f"False positives {label}")
                plt.plot()
                plt.savefig(FP_dir+experiment_name+'.'+str(key)+'.FP_SPREAD.png')
                plt.close('all')
    key_list=df['Label'].unique()
    for colna in ['FP', 'total_fp_miss', 'fp_class_count']:
        key_list_not_covered=set(key_list)
        owi=""
        print(eval_dict_columns)
        if colna not in eval_dict_columns:
            eval_dict_columns.append(colna)
        else:
            owi=eval_dict_columns.index(colna)
        print(eval_dict_columns)
        for key in eval_dict.keys():
            i=eval_dict[key][0]
            if i in key_list:
                bv=eval(colna+'['+str(i)+']')
                key_list_not_covered.remove(i)
                if owi != "":
                    eval_dict[key][owi]=bv
                else:
                    eval_dict[key].append(bv)
        if key_list_not_covered != set():
            for i in list(key_list_not_covered):
                bv=eval(colna+'['+str(i)+']')
                print(f'Warning: {colna} count {bv} for syscall index {i} not added to evaluation dictionary')
                print(f'Warning: {i} does not occur as index in the basic syscall list')
    #
        eval_df=pd.DataFrame.from_dict(eval_dict, orient='index',columns=eval_dict_columns)
        #Add metrics
        eval_df['ACC']=(eval_df['TP']+eval_df['TN'])/(eval_df['TP']+eval_df['FP']+eval_df['FN']+eval_df['TN'])
        eval_df['ACC']=eval_df['ACC'].round(3)
        print('ACC added to eval_dict')
        eval_df['ERR']=(eval_df['FP']+eval_df['FN'])/(eval_df['TP']+eval_df['FP']+eval_df['FN']+eval_df['TN'])
        eval_df['ERR']=eval_df['ERR'].round(3)
        print('ERR added to eval_dict')
        eval_df['PRE']=eval_df['TP']/(eval_df['TP']+eval_df['FP'])
        eval_df['PRE']=eval_df['PRE'].round(3)
        print('PRE added to eval_dict')
        eval_df['REC']=eval_df['TP']/(eval_df['TP']+eval_df['FN'])
        eval_df['REC']=eval_df['REC'].round(3)
        print('REC added to eval_dict')
        eval_df['TNR']=eval_df['TN']/(eval_df['FP']+eval_df['TN'])
        eval_df['TNR']=eval_df['TNR'].round(3)
        print('TNR added to eval_dict')
        eval_df['FPR']=eval_df['FP']/(eval_df['FP']+eval_df['TN'])
        eval_df['FPR']=eval_df['FPR'].round(3)
        print('FPR added to eval_dict')
        eval_df['FNR']=eval_df['FN']/(eval_df['TP']+eval_df['FN'])
        eval_df['FNR']=eval_df['FNR'].round(3)
        print('FNR added to eval_dict')
        eval_df['F1']=2*(eval_df['PRE']*eval_df['REC'])/(eval_df['PRE']+eval_df['REC'])
        eval_df['F1']=eval_df['F1'].round(3)
        print('F1 added to eval_dict')
        Z=eval_df['TP']*eval_df['TN']-eval_df['FP']*eval_df['FN']
        N=(eval_df['TP']+eval_df['FP'])*(eval_df['TP']+eval_df['FN'])*(eval_df['TN']+eval_df['FP'])*(eval_df['TN']+eval_df['FN'])
        N=N.pow(0.5)                                                                                        
        eval_df['MCC']=Z/N
        eval_df['MCC']=eval_df['MCC'].round(3)
        print('MCC added to eval_dict')
        #Kappa = 2 * (TP * TN - FN * FP) / (TP * FN + TP * FP + 2 * TP * TN + FN^2 + FN * TN + FP^2 + FP * TN)
        Z=2*(eval_df['TP']*eval_df['TN']-eval_df['FP']*eval_df['FN'])
        N=eval_df['TP']*eval_df['FN']+eval_df['TP']*eval_df['FP']+2*eval_df['TP']*eval_df['TN']+eval_df['FN'].pow(2)+eval_df['FN']*eval_df['TN']+eval_df['FP'].pow(2)+eval_df['FP']*eval_df['TN']
        eval_df['KAPPA']=Z/N
        eval_df['KAPPA']=eval_df['KAPPA'].round(3)
        print('KAPPA added to eval_dict')
        #
        eval_df=eval_df.convert_dtypes()
        eval_df['MCC']=eval_df['MCC'].round(3)
        eval_df.rename(columns={"total_fn_miss": "FNTOTMIS","total_fp_miss": "FPTOTMIS","fn_class_count": "#FNCLAS","fp_class_count": "#FPCLAS"},inplace=True)
        #
        eval_df.to_csv(experiment_name+'.eval_df.csv')
    results_dict= {}
    print('generating sklearn class report ...')
    class_report_dict=sklearn.metrics.classification_report(df['KEY'],df['Label'],output_dict=True,zero_division=0)
    results_dict['accuracy']=[class_report_dict.pop('accuracy')]
    for x in class_report_dict['macro avg'].keys():
        results_dict['macro_avg'+'_'+x]=[class_report_dict['macro avg'][x]]  
    for x in class_report_dict['weighted avg'].keys():
        results_dict['weighted_avg'+'_'+x]=[class_report_dict['weighted avg'][x]]
    results_dict['MCC']=[sklearn.metrics.matthews_corrcoef(df['KEY'],df['Label'])]
    results_dict['KAPPA']=[sklearn.metrics.cohen_kappa_score(df['KEY'],df['Label'])] 
    class_report_df=pd.DataFrame.from_dict(class_report_dict,orient='index')
    class_report_df['SCNA']=class_report_df.index
    class_report_df['SCNA'] = pd.to_numeric(class_report_df['SCNA'],errors='coerce')
    class_report_df['SCNA'].replace(nr2sc,inplace=True) 
    class_report_df.to_csv(experiment_name+'.final_class_report.csv')
    print(sklearn.metrics.classification_report(df['KEY'],df['Label'],zero_division=0))
    print(results_dict)
    results_df=pd.DataFrame.from_dict(results_dict)
    results_df.to_csv(experiment_name+'.final_result.csv')
    tic = time.perf_counter()
    print(f'Creating CM using the finialized RandomForrestClassifier and the unseen test data {testdata_path} ...')
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(df['KEY'],df['Label'],normalize='true')
    print(cm)   
    F=np.zeros(cm.shape,dtype=bool)
    T=np.ones(cm.shape,dtype=bool)
    plt.rcParams["figure.figsize"]=(30,30)
    cm_cmap='Blues'
    ax=sns.heatmap(cm,annot=False,cmap=cm_cmap)
    #ax=sns.heatmap(cm,mask=np.where(cm==0,F,T),annot=False,cmap=cm_cmap)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.plot()
    plt.savefig(experiment_name+'.CM.FINAL_TEST.png')
    plt.close()
    pd.DataFrame(data=cm).to_hdf(experiment_name+'.CM.FINAL_TEST.h5', key='CM',mode='w',complib='blosc:lz4',complevel=9)
    toc = time.perf_counter()
    print(f'Creation of CM using the finialized RandomForrestClassifier and the unseen test data {testdata_path} took {toc - tic:0.4f} seconds')
    return eval_df

# %%
def run_group_test(testdata_path,model_path,nfeat,reduce_number_of_test_features,accept_test_fna,CNR2NR,NR2SC,tana,load_model_path):
    if not accept_test_fna:
        print(f'{testdata_path} has not been accepted for test, see above')
        return
    if not not_missing_or_empty(model_path):
        print(f'{model_path} is empty or missing')
        print('Warning: no final evaluation will be performed')
        return
    if(load_model_path):
        if load_model_path.endswith('.pkl'):
            model_fna=load_model_path[:-len('.pkl')]
        print(f'Loading model  {model_fna} ...')
        model=load_model(model_fna)
    else:
        if model_path.endswith('.pkl'):
            model_fna=model_path[:-len('.pkl')]
        print(f'Loading model  {model_fna} ...')
        model=load_model(model_fna)
    print(f'{model_fna} loaded')
    if not not_missing_or_empty(testdata_path):
        print(f'{testdata_path} is empty or missing')
        print('Warning: no final evaluation will be performed')
        return
    with h5py.File(testdata_path,'r') as hdf:
        ls=list(hdf.keys())
        non_scs=False
        fail_dict={}
        print(f'Generating {tana} ...')
        if not_missing_or_empty(tana):
            os.remove(tana)
            print(f"Existing {tana} removed")
        for g in ls:
            s=g.split('_')
            if(len(s)!=10):
                continue
            print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv')
            if s[1]==str(65535):
                non_scs=True
            tic = time.perf_counter()
            print(f"Assessing prediction accuracy for group {g} ...")
            ticr = time.perf_counter()
            print(f"Reading group {g} ...")
            df=pd.read_hdf(testdata_path,key=g)
            df=df.iloc[:,0:nfeat]
            df.drop_duplicates(subset=None,ignore_index=True,inplace=True)
            df.reset_index(inplace=True, drop=True)
            df['KEY']=df['KEY'].astype(int)
            tocr = time.perf_counter()
            print(f"Reading group {g} took {tocr - ticr:0.4f} seconds")
            #
            ticp = time.perf_counter()
            print(f'Predicting labels for group {g} ...')
            y_predicted=predict_model(model,data=df)
            df=y_predicted.filter(items=['KEY','Label','Score'])
            print(df.info())
            y_label=list(df['Label'])
            y_label_set=set(y_label)
            size_y_label_set=len(y_label_set)
            print(f"Size of the predicted label set for {s[1]} is {size_y_label_set}:")
            #transform y_label back to non-compact representation
            y_label_not_compact=[] 
            for sci in y_label:
                bv=CNR2NR.get(str(sci))
                if bv==None:
                    bv=65535
                    CNR2NR[sci]=str(bv)
                y_label_not_compact.append(int(bv))
            y_label=y_label_not_compact.copy()
            del(y_label_not_compact) 
            df['Label']=y_label
            y_score=list(df['Score'])
            y_key=list(df['KEY'])
            y_key_set=set(y_key)
            if len(y_key_set)!=1:
                print(f'Warning: key set for {s[1]} does not contain exactly one element')
                print(f"Warning: size of the true label set for {s[1]} is {len(y_key_set)}:")
                #print(y_key_set)
            else:
                print(f'OK: true label set of group {g} contains exactly one element')
                if int(list(y_key_set)[0]) != int(s[1]):
                    print(f'Warning: true label set for {s[1]} contains {s[1]} as single element but should contain {list(y_key_set)[0]} instead')
                else:
                    print(f'OK: this single element is {int(s[1])}')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(g)
            print(df)
            df.to_hdf(tana,key='KEY_LABEL_SCORE',mode='a',format='table',append=True,complib='blosc:lz4',complevel=9)
            del(df)
            del(y_predicted)
            #
            tocp = time.perf_counter()
            print(f"Label prediction for group {g} took {tocp - ticp:0.4f} seconds")
            total=len(y_label)
            success=len([i for i,j in zip(y_key,y_label) if i == j])
            print(f'group {g}: {total-success}/{total} predictions failed')
            toc = time.perf_counter()
            print(f"Assessment of prediction accuray for group {g} took {toc - tic:0.4f} seconds")
            print("")
    #
    eval_df=final_evaluation(tana,eval_dict_columns,eval_dict,testdata_path)
    return eval_df

# %%
failure=[]
success=[]
missing=[]
F=0
#
#Book keeping files
success_na='./SUCCESS.'+config_hash+'.csv'
if not_missing_or_empty(success_na):
    os.remove(success_na)
failure_na='./FAILURE.'+config_hash+'.csv'
if not_missing_or_empty(failure_na):
    os.remove(failure_na)
plot_failure_na='./PLOT_FAILURE.'+config_hash+'.csv'
#if not_missing_or_empty(plot_failure_na):
#    os.remove(plot_failure_na)
if os.path.exists(plot_failure_na):
        sh.rm('f', plot_failure_na)
tune_failure_na='./TUNE_FAILURE.'+config_hash+'.csv'
if not_missing_or_empty(tune_failure_na):
    os.remove(tune_failure_na)
missing_na='./MISSING.'+config_hash+'.csv'
if not_missing_or_empty(missing_na):
    os.remove(missing_na)
shape_na='./SHAPE_LOG.'+config_hash+'.csv'
if not_missing_or_empty(shape_na):
    os.remove(shape_na)
#
TIC = time.perf_counter()
#
for channel in l_channel:
    for blur in l_blur:
        test_fna=''
        test_nfeat_set=set()
        test_nfeat=0
        bv=str(channel)+'_'+str(blur)
        if test_file_dict.get(bv) != None:
            test_fna=os.path.realpath(test_file_dict[bv])
            if not_missing_or_empty(test_fna):
                print(f'{test_fna} contains test data for channel={channel} and blur={blur}')
                #determine test_dict
                test_shape_dict={}
                print(f"Analyzing {test_fna} ...")
                tic = time.perf_counter()
                with h5py.File(test_fna,'r') as hdf:
                    ls=list(hdf.keys())
                    non_scs=False
                    for g in ls:
                        s=g.split('_')
                        if(len(s)!=10):
                            continue
                        if s[1]==str(65535):
                            non_scs=True
                        c=0
                        for dset in hdf[g].keys():
                            if c==1:
                                arr=hdf[g][dset] 
                                x=arr.shape
                                x=x[0] 
                                exec('y='+str(hdf[g][dset].dtype[-1]))
                                y=y[1][0]
                                if y==1:
                                    exec('y='+str(hdf[g][dset].dtype[-2]))                                    
                                test_shape_dict[g]=(x,y)
                            c+=1
                        print(g,":",test_shape_dict[g])
                        #print(g,":",test_shape_dict[g],df.duplicated().sum())
                    if len(test_shape_dict)==0:
                            print(f'{test_fna} has no identifiable syscall blocks')
                            test_fna=""
                    else:
                        if non_scs==False:
                            print(f'Warning: {test_fna} contains no non-syscalls')
                        for g in test_shape_dict.keys():
                            test_nfeat_set.add(test_shape_dict[g][1])
                        if len(test_nfeat_set)!=1:
                            print(f'Warning: The syscall goups in {test_fna} have not all the same nfeat value ')
                            print(f'{test_fna} is rejected as test file')
                            test_fna=""
                        else:
                            test_nfeat=list(test_nfeat_set)
                            test_nfeat=test_nfeat[-1]+1
                            print(f'{test_fna} has {test_nfeat} features and {len(test_shape_dict)} syscall blocks  ')
                    toc = time.perf_counter()
                    print(f"Analysis of {test_fna}  took {toc - tic:0.4f} seconds")
            else:
                print(f'Warning: {test_fna} provided as test file for channel={channel} and blur={blur} is empty or missing')
                print(f'Warning: No extra test data are available for all nfeats of channel={channel} and blur={blur}')
                test_fna=''
        for nfeat in l_nfeat:
            ticc = time.perf_counter()
            accept_test_fna=False
            reduce_number_of_test_features=False
            if test_fna!='':
                if test_nfeat==0:
                    print(f'Error: test_nfeat should be > 0 at this stage, but is not')
                else:
                    if nfeat > test_nfeat:
                        print(f'Warning: train nfeat={nfeat} > test nfeat={test_nfeat}')
                        print(f'Warning: cannot use {test_fna}')
                    else:
                        accept_test_fna=True
                        if nfeat < test_nfeat:
                             reduce_number_of_test_features=True 
            shape_count=0
            print("------------------------------------------------------------")
            F+=1
            na=channel+"_"+str(nfeat)+"_"+str(blur)
            fna=h5_path+na+"_0.h5"
            experiment_name=na+'_'+model_name+experiment_name_suffix
            if not_missing_or_empty(fna):
                load_model_path=""
                if load_model_dict.get(na) != None:
                    load_model_path=os.path.realpath(load_model_dict.get(na))
                    if not not_missing_or_empty(load_model_path):
                        print(f'Warning: {load_model_path} empty or missing')
                        load_model_path=""
                if load_model_path=="":
                    print(f"Training {model_name} for group {na} ...")
                else:
                    print(f'Will skip model training and load existing model path {load_model_path} instead')
                try:
                    ticr = time.perf_counter()
                    print(f'Training model for group {na} ...')
                    path_to_final_model=run_group_training(na,fna,experiment_name,size_cut,n_trials,use_existing_feature_rejections,load_model_path)
                except Exception as e:
                    print(e)
                    failure.append([channel,nfeat,blur,e])
                    df_failure = pd.DataFrame(failure,columns=['CHANNEL', 'NFEAT', 'BLUR','ERROR_MESSAGE'])
                    df_failure.to_csv(failure_na)
                    print(f"Training of {model_name} for group {na} failed")
                    tocr = time.perf_counter()
                    print(f"Training for group {na} took {tocr - ticr:0.4f} seconds")
                else:
                    success.append([channel,nfeat,blur] )
                    df_success = pd.DataFrame(success,columns=['CHANNEL', 'NFEAT', 'BLUR'])
                    df_success.to_csv(success_na)
                    print(f"Training of {model_name} for group {na} succeeded")
                    tocr = time.perf_counter()
                    print(f"Training for group {na} took {tocr - ticr:0.4f} seconds")
                    #
                    allow_final_test_evaluation=True
                    if allow_final_test_evaluation:
                        if test_fna!="":
                            if path_to_final_model=='':
                                print(f'Warning: skipping final test for {test_fna}, since no path to finalized modell is available')
                            elif not not_missing_or_empty(path_to_final_model):
                                print(f'Warning: skipping final test for {test_fna}, finalized model {path_to_final_model} is empty or missing')
                            else:
                                tict = time.perf_counter()
                                print(f'Running final test for {path_to_final_model} with test data from {test_fna} ...')
                                CNR2NR_NA=experiment_name+'.CNR2NR.json'
                                NR2SC_NA='./NR2SC.json'
                                if not not_missing_or_empty(CNR2NR_NA):
                                    print(f'{CNR2NR_NA} empty or missing')
                                    print(f'Warning: CNR2NR dictionary required for back transforming model syscall indices to non-compact form')
                                    print(f'Warning: test of group {na} will be skipped')
                                elif not not_missing_or_empty(NR2SC_NA):
                                    print(f'{NR2SC_NA} empty or missing')
                                    print(f'Warning: NR2SC dictionary required for back transforming model syscall indices to syscall names')
                                    print(f'Warning: test of group {na} will be skipped')
                                else:
                                    CNR2NR={}
                                    NR2SC={}
                                    with open(CNR2NR_NA) as f: 
                                        CNR2NR=json.load(f)
                                    with open(NR2SC_NA) as f: 
                                       NR2SC=json.load(f)
                                    tana=experiment_name+'.KEY_LABEL_SCORE.h5'
                                    run_group_test(test_fna,path_to_final_model,nfeat,reduce_number_of_test_features,accept_test_fna,CNR2NR,NR2SC,tana,load_model_path)
                                    toct = time.perf_counter()
                                    print(f"Final test for group {na}: {toct - tict:0.4f} seconds")
                        else: 
                            print('No final test file available')
                            print('Warning: final test evaluation is skipped')
                    else:
                        print('Final test evaluation switch off by configuration')
            else:
                missing.append([channel,nfeat,blur] )
                df_missing = pd.DataFrame(missing,columns=['CHANNEL', 'NFEAT', 'BLUR'])
                df_missing.to_csv(missing_na)
                print(f"Warning: {fna} empty or missing")
            tocc = time.perf_counter()
            print(f"Total training and evaluation time for group {na}: {tocc - ticc:0.4f} seconds")

print("------------------------------------------------------------")                
TOC = time.perf_counter()
print(f"Total training and evaluation time for all groups: {TOC - TIC:0.4f} seconds")
print("")

# %%
if len(success)>0:
    print(f"{len(success)}/{F} trainings succeeded")
if len(failure)>0:
    print(f"{len(failure)}/{F} trainings failed")
if len(missing)>0:
    print(f"{len(missing)}/{F} trainings without input data")
accounted=len(success)+len(failure)+len(missing)
if accounted<F:
    print(f"Warning: {accounted}/{F} trainings cannot be accounted for")
if accounted>F:
    print(f"Warning: {accounted}/{F} trainings accounted for, but only {F} available")

# %%
case_na="_"+model_name+"."+config_hash
li=[] 
for channel in l_channel:
    for blur in l_blur:
        for nfeat in l_nfeat:
            group_na=channel+"_"+str(nfeat)+"_"+str(blur)
            fna=group_na+case_na+".results.csv"
            c=0
            if not_missing_or_empty(fna):
                c+=1
                df=pd.read_csv(fna).reset_index(drop=True)
                data=[[channel,nfeat,blur,n] for n in range(df.shape[0])]
                df=pd.concat([pd.DataFrame(data, columns=['CHANNEL','NFEAT','BLUR','FOLD']),df],axis=1)              
                li.append(df)
df = pd.concat(li,ignore_index=True)
print(df)
df.to_csv('results'+case_na+'.csv',index=0)
#
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3,3,figsize=(15, 15), sharex=True, sharey=True)
fig.suptitle('Accuracy of Random Forrest Classifier')
for ax in fig.get_axes():
    ax.label_outer()
ax1.set_title("PR BLUR=0")
sns.lineplot(ax=ax1, data=df.loc[ (df['CHANNEL']=='pr') & (df['BLUR']==0) & (df['FOLD']<10) ],x='NFEAT',hue='FOLD',y='Accuracy',markers=True,)
ax2.set_title("PR BLUR=4")
sns.lineplot(ax=ax2, data=df.loc[ (df['CHANNEL']=='pr') & (df['BLUR']==4) & (df['FOLD']<10) ],x='NFEAT',hue='FOLD',y='Accuracy',markers=True,)
ax3.set_title("PR BLUR=12")
sns.lineplot(ax=ax3, data=df.loc[ (df['CHANNEL']=='pr') & (df['BLUR']==12) & (df['FOLD']<10) ],x='NFEAT',hue='FOLD',y='Accuracy',markers=True,)

ax4.set_title("PW BLUR=0")
sns.lineplot(ax=ax4, data=df.loc[ (df['CHANNEL']=='pw') & (df['BLUR']==0) & (df['FOLD']<10) ],x='NFEAT',hue='FOLD',y='Accuracy',markers=True,)
ax5.set_title("PW BLUR=4")
sns.lineplot(ax=ax5, data=df.loc[ (df['CHANNEL']=='pw') & (df['BLUR']==4) & (df['FOLD']<10) ],x='NFEAT',hue='FOLD',y='Accuracy',markers=True,)
ax6.set_title("PW BLUR=12")
sns.lineplot(ax=ax6, data=df.loc[ (df['CHANNEL']=='pw') & (df['BLUR']==12) & (df['FOLD']<10) ],x='NFEAT',hue='FOLD',y='Accuracy',markers=True,)

ax7.set_title("MA BLUR=0")
sns.lineplot(ax=ax7, data=df.loc[ (df['CHANNEL']=='ma') & (df['BLUR']==0) & (df['FOLD']<10) ],x='NFEAT',hue='FOLD',y='Accuracy',markers=True,)
ax8.set_title("MA BLUR=4")
sns.lineplot(ax=ax8, data=df.loc[ (df['CHANNEL']=='ma') & (df['BLUR']==4) & (df['FOLD']<10) ],x='NFEAT',hue='FOLD',y='Accuracy',markers=True,)
ax9.set_title("MA BLUR=12")
sns.lineplot(ax=ax9, data=df.loc[ (df['CHANNEL']=='ma') & (df['BLUR']==12) & (df['FOLD']<10) ],x='NFEAT',hue='FOLD',y='Accuracy',markers=True,)
fig.tight_layout()
plt.plot()
plt.savefig('accuracy_over_nfeat.'+config_hash+'.png')
plt.close()


