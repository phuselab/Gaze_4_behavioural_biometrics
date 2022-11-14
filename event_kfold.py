from os.path import join
import scipy
import os
import numpy as np
from my_utils.loader import load_event_features
from sklearn.preprocessing import label_binarize, StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, plot_confusion_matrix
import numpy_indexed as npi
from sklearn.svm import SVC
from scipy.stats import uniform
import pandas as pd
import re
from my_utils.plotter import build_roc_curve

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key = alphanum_key)


def train_sklearn(X, y, model='SVM', hyper_search=True):
    from sklearn.metrics import make_scorer, f1_score
    scorer = make_scorer(f1_score, average='macro')

    pipe_svc = make_pipeline(RobustScaler(),
                             SVC(random_state=1, C=1000, gamma=0.002, kernel='rbf'))
    distributions = dict(svc__C=scipy.stats.expon(scale=1000), svc__gamma=scipy.stats.expon(scale=.1))
        
    gs = RandomizedSearchCV(pipe_svc,
                            distributions,
                            scoring='accuracy',
                            n_iter=10,
                            n_jobs=-1,
                            cv=5)
    if hyper_search:
        gs = gs.fit(X, y)
        print('Best parameters: ', gs.best_params_)
        score = gs.score(X, y)
        print('\tAccuracy: ' + str(score))
        clf = gs.best_estimator_
        return clf
    else:
        pipe_svc = pipe_svc.fit(X, y)
        score = pipe_svc.score(X, y)
        print('\tAccuracy: ' + str(score))
        return pipe_svc


def score_fusion(clf_fix, clf_sac, X_fix_test, y_f_test, stim_f_test, X_sac_test, y_s_test, stim_s_test, model='SVM'):
    #Fixations -------
    ss = np.zeros_like(y_f_test).astype('str')
    for i in range(len(y_f_test)):
        ss[i] = str(int(y_f_test[i])) + '-' + str(int(stim_f_test[i]))
    if model == 'SVM':
        ppred_fix = clf_fix.decision_function(X_fix_test)
    elif model == 'GP':
        ppred_fix, _ = clf_fix.predict_y(X_fix_test)
        #ppred_fix, _ = clf_fix.predict_f(X_fix_test)
        ppred_fix = ppred_fix.numpy()
    elif model == 'NN':
        ppred_fix = clf_fix.predict(X_fix_test)
    else:
        ppred_fix = clf_fix.predict_proba(X_fix_test)
    key, ppred_fix_comb = npi.group_by(ss).mean(ppred_fix)
    y_test = np.zeros(key.shape)
    for i,k in enumerate(key):
        l = int(k.split('-')[0])
        y_test[i] = l
    
    #Saccades -------
    ss = np.zeros_like(y_s_test).astype('str')
    for i in range(len(y_s_test)):
        ss[i] = str(int(y_s_test[i])) + '-' + str(int(stim_s_test[i]))
    if model == 'SVM':
        ppred_sac = clf_sac.decision_function(X_sac_test)
    elif model == 'GP':
        ppred_sac, _ = clf_sac.predict_y(X_sac_test)
        #ppred_sac, _ = clf_sac.predict_f(X_sac_test)
        ppred_sac = ppred_sac.numpy()
    elif model == 'NN':
        ppred_sac = clf_fix.predict(X_fix_test)
    else:
        ppred_sac = clf_sac.predict_proba(X_sac_test)
    #Fusion --------
    _, ppred_sac_comb = npi.group_by(ss).mean(ppred_sac)
    ppred = np.asarray((np.matrix(ppred_fix_comb) + np.matrix(ppred_sac_comb)) / 2.)
    #ppred = np.asarray((np.matrix(ppred_fix_comb)))
    y_pred = np.squeeze(np.asarray(ppred.argmax(axis=1)))

    #import code; code.interact(local=locals())

    return y_test.astype(int), y_pred, ppred

def load_dataset(path, nsub=None, num_sessions=None):
    global_data_fix = []
    global_data_sac = []
    subs = sorted_nicely(os.listdir(path))
    if nsub is not None:
        subs = subs[:nsub]
    subs_considered = 0
    for file in subs:
        if file == '.DS_Store':
            continue

        fix_data, sac_data, stim_fix, stim_sac = load_event_features(join(path, file))
    
        if num_sessions is not None:
            ns = len(np.unique(stim_fix))
            if ns < num_sessions:
                continue
        label = int(file.split("_")[2].split(".")[0])        
        curr_label_f = np.ones([fix_data.shape[0], 1]) * label
        curr_label_s = np.ones([sac_data.shape[0], 1]) * label
        fix_data = np.hstack([curr_label_f, stim_fix, fix_data])
        sac_data = np.hstack([curr_label_s, stim_sac, sac_data])
        global_data_fix.append(fix_data)
        global_data_sac.append(sac_data)
        subs_considered += 1
    data_fix = np.vstack(global_data_fix)
    data_sac = np.vstack(global_data_sac)
    print('\nLoaded ' + str(subs_considered) + ' subjects...')
    return data_fix, data_sac

def get_CV_splits(stim_f, yf, k):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k)
    subs_splits = []
    sub_labels = np.unique(yf)
    for s in sub_labels:
        curr_stims = np.unique(stim_f[yf==s])[:,np.newaxis]
        subs_splits.append(kf.split(curr_stims))
    return subs_splits, sub_labels

def get_results_kfold(X_fix, yf, stim_f, X_sac, ys, stim_s, k, model='SVM', hyper_search=True, feat_type='OU'):
    sub_splits_gen, sub_labels = get_CV_splits(stim_f, yf, k=k)
    sub_splits = {}
    for i,ss in enumerate(sub_splits_gen):
        curr_splits = []
        for train_index, test_index in ss:
            curr_splits.append((train_index, test_index))
        sub_splits[sub_labels[i]] = curr_splits
    acc_scores = []
    eer_scores = []
    f1_scores = []
    auc_scores = []
    for fold in range(k):
        print('\nFold ' + str(fold+1) + ' of ' + str(k))
        train_Xf = []
        train_yf = []
        train_Xs = []
        train_ys = []
        test_Xf = []
        test_yf = []
        test_Xs = []
        test_ys = []
        train_stf = []
        train_sts = []
        test_stf = []
        test_sts = []
        for s in sub_splits.keys():
            curr_Xf = X_fix[yf==s,:]
            curr_stf = stim_f[yf==s]
            curr_Xs = X_sac[ys==s,:]
            curr_sts = stim_s[ys==s]
            train_index = sub_splits[s][fold][0]
            test_index = sub_splits[s][fold][1]
            for ti in train_index:
                train_Xf.append(curr_Xf[curr_stf==ti])
                train_stf.append(curr_stf[curr_stf==ti])
                train_yf.append(np.repeat(s, len(train_stf[-1])))
                train_Xs.append(curr_Xs[curr_sts==ti])
                train_sts.append(curr_sts[curr_sts==ti])
                train_ys.append(np.repeat(s, len(train_sts[-1])))
            for ti in test_index:
                test_Xf.append(curr_Xf[curr_stf==ti])
                test_stf.append(curr_stf[curr_stf==ti])
                test_yf.append(np.repeat(s, len(test_stf[-1])))
                test_Xs.append(curr_Xs[curr_sts==ti])
                test_sts.append(curr_sts[curr_sts==ti])
                test_ys.append(np.repeat(s, len(test_sts[-1])))
        train_Xf = np.vstack(train_Xf)
        train_yf = np.concatenate(train_yf)
        train_stf = np.concatenate(train_stf)
        train_Xs = np.vstack(train_Xs)
        train_ys = np.concatenate(train_ys)
        train_sts = np.concatenate(train_sts)
        test_Xf = np.vstack(test_Xf)
        test_yf = np.concatenate(test_yf)
        test_stf = np.concatenate(test_stf)
        test_Xs = np.vstack(test_Xs)
        test_ys = np.concatenate(test_ys)
        test_sts = np.concatenate(test_sts)

        print('\nTraining Fixations (SVM)')
        clf_fix = train_sklearn(train_Xf, train_yf, model=model, hyper_search=hyper_search)
        print('Training Saccades (SVM)')
        clf_sac = train_sklearn(train_Xs, train_ys, model=model, hyper_search=hyper_search)

        y_test, y_pred_test, y_score = score_fusion(clf_fix, clf_sac, 
                                                    test_Xf, 
                                                    test_yf, test_stf, 
                                                    test_Xs, 
                                                    test_ys, test_sts, 
                                                    model=model)

        f1score = f1_score(y_true=y_test, y_pred=y_pred_test, average='macro')
        acc_score = accuracy_score(y_true=y_test, y_pred=y_pred_test)
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        auc, eer, _, _ = build_roc_curve(y_test_bin, y_score, max(np.unique(y_test))+1, None, None, show=False)

        print('\nTest Accuracy Score: ' + str(f1score))
        acc_scores.append(acc_score)
        f1_scores.append(f1score) 
        auc_scores.append(auc)
        eer_scores.append(eer)

    return np.mean(acc_scores), np.std(acc_scores), np.mean(f1_scores), np.std(f1_scores), np.mean(auc_scores), np.std(auc_scores), np.mean(eer_scores), np.std(eer_scores)


# MAIN ---------------------------------------------------------------------

dataset_name = 'FIFA_OU_posterior_VI'

feat_type = 'OU'
model = 'SVM'

print('\n\tCERF Dataset (OU features)...\n')
train_dir = join(join('features', dataset_name), 'train')
test_dir = join(join('features', dataset_name), 'test')
data_fix_train, data_sac_train = load_dataset(train_dir)
data_fix_test, data_sac_test = load_dataset(test_dir)
data_fix = np.vstack([data_fix_train, data_fix_test])
data_sac = np.vstack([data_sac_train, data_sac_test])

X_fix = data_fix[:, 2:]
yf = data_fix[:, 0]
stim_f = data_fix[:, 1]
X_sac = data_sac[:, 2:]
ys = data_sac[:, 0]
stim_s = data_sac[:, 1]

n_class_f = len(np.unique(yf))
n_class_s = len(np.unique(ys))
assert n_class_f == n_class_s

print('\nNumber of classes: ' + str(n_class_f))

unique_f, counts_f = np.unique(yf, return_counts=True)
cf = dict(zip(unique_f.astype(int), counts_f))

unique_s, counts_s = np.unique(ys, return_counts=True)
cs = dict(zip(unique_s.astype(int), counts_s))

print('\n-------------------------------')
print('\nFixations Counts per Class: \n' + str(cf))
print(' ')
print('Saccades Counts per Class: \n' + str(cs))
print('\n-------------------------------')

acc_score, acc_std, f1_score, f1_std, auc_score, auc_std, eer_score, eer_std = get_results_kfold(X_fix, yf, stim_f, X_sac, ys, stim_s, k=10, model=model, hyper_search=True, feat_type=feat_type)

print('\nAccuracy CV score: ' + str(acc_score) + ' +- ' + str(acc_std))
print('F1 CV score: ' + str(f1_score) + ' +- ' + str(f1_std))
print('AUC CV score: ' + str(auc_score) + ' +- ' + str(auc_std))
print('EER CV score: ' + str(eer_score) + ' +- ' + str(eer_std))
print(' ')