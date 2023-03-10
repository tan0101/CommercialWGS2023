# -*- coding: utf-8 -*-
"""
Created on Fri Sep  24 20:35:00 2021

@author: Alexandre Maciel Guerra
"""

import numpy as np
from numpy.core.numeric import indices
from numpy.lib.index_tricks import s_
import pandas as pd
import sys
import os
import pickle

from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from collections import Counter
from sklearn.model_selection import StratifiedKFold, GridSearchCV,cross_validate, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
#from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from pathlib import Path
#from sklearn.pipeline import Pipeline


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category= FutureWarning)
simplefilter(action='ignore', category= UserWarning)
simplefilter(action='ignore', category= DeprecationWarning)

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
           'fp': make_scorer(fp), 'fn': make_scorer(fn),
           'auc': 'roc_auc',
           'acc': make_scorer(accuracy_score),
           'kappa': make_scorer(cohen_kappa_score)}

def update_progress(progress):
    barLength = 100 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100, 2), status)
    sys.stdout.write(text)
    sys.stdout.flush()  

def pre_processing_selection(name_dataset, data_folder, name_antibiotic, data, features_name):
    # Get features accessory:
    print("Before feature selection:{}".format(data.shape))
    df_weighted_features = pd.read_csv(folder+"/Population Correction/"+data_folder+"/Chi Square Features/"+name_dataset+"_"+name_antibiotic+"_model_pvalue.csv", header = [0], index_col=[0])
    cols=np.array(df_weighted_features[df_weighted_features.columns[0]]).astype(int)
    data = data[:,cols]
    features_anti = features_name[cols]
    print("After population correction:{}".format(data.shape))
    
    
    # Preprocessing accessory - Feature Selection
    std_scaler = StandardScaler()
    data = std_scaler.fit_transform(data)
    model = SelectFromModel(ExtraTreesClassifier(n_estimators=50, random_state=0)).fit(data, target)
    coef = model.estimator_.feature_importances_
    cols_model = model.get_support(indices=True)
    coef_sel = coef[cols_model]
    features_anti = features_anti[cols_model]
    data = data[:, cols_model]
    print("After select from model:{}".format(data.shape))

    return data, coef_sel, features_anti


if __name__ == "__main__":
    name_dataset = "SAL143" 
    folder = "SalmonellaSNPs"
    results_folder = "Results"
    type_data = "accessory_core_intergenic"
     
    # Load Data Core:
    data_core = pd.read_csv(folder+"/"+name_dataset+'_core_genome_snps.csv', header = [0], index_col=[0])
    data_core = data_core.transpose()
    data_txt_core = np.array(data_core)
    features_name_core = np.array(data_core.columns)
    sample_name_core = np.array(data_core.index)
    print(data_txt_core.shape)

    # Load Data Accessory:
    data_acc = pd.read_csv(folder+"/"+name_dataset+'_accessory_genes.csv', header = [0], index_col=[0])
    data_acc = data_acc.transpose()
    data_acc = data_acc.loc[sample_name_core,:]
    data_txt_acc = np.array(data_acc)
    features_name_acc = np.array(data_acc.columns)
    sample_name_acc = np.array(data_acc.index)
    print(data_txt_acc.shape)

    # Load Data Intergenic:
    data_inter = pd.read_csv(folder+"/"+name_dataset+'_intergenic_snps.csv', header = [0], index_col=[0])
    data_inter = data_inter.transpose()
    data_inter = data_inter.loc[sample_name_core,:]
    data_txt_inter = np.array(data_inter)
    features_name_inter = np.array(data_inter.columns)
    sample_name_inter = np.array(data_inter.index)
    print(data_txt_inter.shape)

    # Load Antibiotic Data:
    antibiotic_df = pd.read_csv(folder+"/"+name_dataset+'_AMR_data_RSI.csv', header = [0])
    order_id = []
    human_data_id = []
    
    for count, s_name in enumerate(sample_name_core):
        idx = np.where(antibiotic_df[antibiotic_df.columns[0]] == s_name)[0]
        order_id.append(idx[0])
        
    antibiotic_df = antibiotic_df.iloc[order_id, :].reset_index()
    antibiotic_df.drop(columns="index",axis=1,inplace=True)
    samples = np.array(antibiotic_df[antibiotic_df.columns[0]])
    print("samples ini length = {}".format(len(samples)))

    # Nested Cross Validation:
    inner_loop_cv = 3   
    outer_loop_cv = 5
    
    # Number of random trials:
    NUM_TRIALS = 30
    
    # Grid of Parameters:
    C_grid = {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    est_grid = {"clf__n_estimators": [2, 4, 8, 16, 32, 64]}
    MLP_grid = {"clf__alpha": [0.001, 0.01, 0.1, 1, 10, 100], "clf__learning_rate_init": [0.001, 0.01, 0.1, 1],
        "clf__hidden_layer_sizes": [10, 20, 40, 100, 200, 300, 400, 500]}
    SVC_grid = {"clf__gamma": [0.0001, 0.001, 0.01, 0.1], "clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    DT_grid = {"clf__max_depth": [10, 20, 30, 50, 100]}
    XGBoost_grid = {"clf__n_estimators": [2, 4, 8, 16, 32, 64], "clf__learning_rate": [0.001, 0.01, 0.1, 1]}
        
    # Classifiers:
    names = ["Logistic Regression", "Linear SVM", "RBF SVM",
        "Extra Trees", "Random Forest", "AdaBoost", "XGBoost"]

    classifiers = [
        LogisticRegression(),
        LinearSVC(loss='hinge'),
        SVC(),
        ExtraTreesClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier()
        ]
    
    print(antibiotic_df.columns[1:])
    for name_antibiotic in antibiotic_df.columns[1:]:
        print("Antibiotic: {}".format(name_antibiotic))

        target_str = np.array(antibiotic_df[name_antibiotic])
        
        target = np.zeros(len(target_str)).astype(int)
        idx_S = np.where(target_str == 'S')[0]
        idx_R = np.where(target_str == 'R')[0]
        idx_NaN = np.where((target_str != 'R') & (target_str != 'S'))[0]
        target[idx_R] = 1    

        if len(idx_NaN) > 0:
            target = np.delete(target,idx_NaN)
            print("Correct number of isolates: {}".format(len(target)))
        
        # Check minimum number of samples:
        count_class = Counter(target)
        print(count_class)
        if count_class[0] < 20 or count_class[1] < 20:
            continue 

        
        # Accessory Genes
        data_acc, coef_sel_acc, features_anti_acc = pre_processing_selection(name_dataset, "accessory_genes", name_antibiotic, data_acc, features_name_acc)
        directory = folder+"/Population Correction/"+results_folder+"/accessory_genes"
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        df_features = pd.DataFrame(coef_sel_acc, columns = ["Importance"], index=features_anti_acc)
        df_features.to_csv(directory+"/features_"+name_dataset+"_"+name_antibiotic+".csv")


        # Core Genome SNPs
        data_core, coef_sel_core, features_anti_core = pre_processing_selection(name_dataset, "core_genome_snps", name_antibiotic, data_core, features_name_core)
        directory = folder+"/Population Correction/"+results_folder+"/core_genome_snps"
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        df_features = pd.DataFrame(coef_sel_core, columns = ["Importance"], index=features_anti_core)
        df_features.to_csv(directory+"/features_"+name_dataset+"_"+name_antibiotic+".csv")
        
        # Intergenic Region SNPs
        data_inter, coef_sel_inter, features_anti_inter = pre_processing_selection(name_dataset, "intergenic_snps", name_antibiotic, data_inter, features_name_inter)
        directory = folder+"/Population Correction/"+results_folder+"/intergenic_snps"
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        df_features = pd.DataFrame(coef_sel_inter, columns = ["Importance"], index=features_anti_inter)
        df_features.to_csv(directory+"/features_"+name_dataset+"_"+name_antibiotic+".csv")

        # Combine datasets
        data = np.concatenate((data_acc, data_core, data_inter), axis=0)
                        
        # Initialize Variables:
        scores_auc = np.zeros([NUM_TRIALS,len(classifiers)])
        scores_acc = np.zeros([NUM_TRIALS,len(classifiers)])
        scores_sens = np.zeros([NUM_TRIALS,len(classifiers)])
        scores_spec = np.zeros([NUM_TRIALS,len(classifiers)])
        scores_kappa = np.zeros([NUM_TRIALS,len(classifiers)])
        scores_prec = np.zeros([NUM_TRIALS,len(classifiers)])
        
        # Loop for each trial
        update_progress(0)
        for i in range(NUM_TRIALS):
            #print("Trial = {}".format(i))
        
            inner_cv = StratifiedKFold(n_splits=inner_loop_cv, shuffle=True, random_state=i)
            outer_cv = StratifiedKFold(n_splits=outer_loop_cv, shuffle=True, random_state=i)
        
            k = 0
        
            for name, clf in zip(names, classifiers):
                
                model = Pipeline([('sampling',SMOTEENN(sampling_strategy = "all", random_state=i)),
                                ('clf', clf)])
                
                if name == "QDA" or name == "LDA" or name == "Naive Bayes":
                    classif = model
                else:
                    if name == "RBF SVM":
                        grid = SVC_grid              
                    elif name == "Random Forest" or name == "AdaBoost" or name == "Extra Trees":
                        grid = est_grid
                    elif name == "Neural Net":
                        grid = MLP_grid
                    elif name == "Linear SVM":
                        grid = C_grid
                    elif name == "Decision Tree":
                        grid = DT_grid
                    elif name == "XGBoost":
                        grid = XGBoost_grid
                    else:
                        grid = C_grid

                    # Inner Search
                    classif = GridSearchCV(estimator=model, param_grid=grid, cv=inner_cv)
                    classif.fit(data, target)
                
                # Outer Search
                cv_results = cross_validate(classif, data, target, scoring=scoring, cv=outer_cv, return_estimator=True)

                tp = cv_results['test_tp']
                tn = cv_results['test_tn']
                fp = cv_results['test_fp']
                fn = cv_results['test_fn']
                
                sens = np.zeros(outer_loop_cv)
                spec = np.zeros(outer_loop_cv)
                prec = np.zeros(outer_loop_cv)
                
                for j in range(outer_loop_cv):
                    TP = tp[j]
                    TN = tn[j]
                    FP = fp[j]
                    FN = fn[j]
                    
                    # Sensitivity, hit rate, recall, or true positive rate
                    sens[j] = TP/(TP+FN)
                    
                    # Fall out or false positive rate
                    FPR = FP/(FP+TN)
                    spec[j] = 1 - FPR
                    if TP + FP > 0:
                        prec[j] = TP / (TP + FP)
    
                scores_sens[i,k] = sens.mean()
                scores_spec[i,k] = spec.mean()
                scores_prec[i,k] = prec.mean()
                scores_auc[i,k] = cv_results['test_auc'].mean()
                scores_acc[i,k] = cv_results['test_acc'].mean()
                scores_kappa[i,k] = cv_results['test_kappa'].mean()
                
                k = k + 1
                
            update_progress((i+1)/NUM_TRIALS)

        results = np.zeros((12,len(classifiers)))
        scores = [scores_auc, scores_acc, scores_sens, scores_spec, scores_kappa, scores_prec]
        for counter_scr, scr in enumerate(scores):
            results[2*counter_scr,:] = np.mean(scr,axis=0)
            results[2*counter_scr + 1,:] = np.std(scr,axis=0)
            
        names_scr = ["AUC_Mean", "AUC_Std", "Acc_Mean", "Acc_Std", 
            "Sens_Mean", "Sens_Std", "Spec_Mean", "Spec_Std", 
            "Kappa_Mean", "Kappa_Std", "Prec_Mean", "Prec_Std"]

        results_df=pd.DataFrame(results, columns=names, index=names_scr)
        
        directory = folder+"/Population Correction/"+results_folder+"/"+type_data
        if not os.path.exists(directory):
            os.makedirs(directory)

        results_df=pd.DataFrame(results, columns=names, index=names_scr)
        results_df.to_csv(directory+"/SMOTEENN_results_"+name_dataset+"_"+name_antibiotic+".csv")

        df_auc = pd.DataFrame(scores_auc, columns=names)
        df_auc.to_csv(directory+"/SMOTEENN_"+name_dataset+"_"+name_antibiotic+"_auc.csv")
        
        df_acc = pd.DataFrame(scores_acc, columns=names)
        df_acc.to_csv(directory+"/SMOTEENN_"+name_dataset+"_"+name_antibiotic+"_acc.csv")
        
        df_sens = pd.DataFrame(scores_sens, columns=names)
        df_sens.to_csv(directory+"/SMOTEENN_"+name_dataset+"_"+name_antibiotic+"_sens.csv")
        
        df_spec = pd.DataFrame(scores_spec, columns=names)
        df_spec.to_csv(directory+"/SMOTEENN_"+name_dataset+"_"+name_antibiotic+"_spec.csv")
        
        df_kappa = pd.DataFrame(scores_kappa, columns=names)
        df_kappa.to_csv(directory+"/SMOTEENN_"+name_dataset+"_"+name_antibiotic+"_kappa.csv")
        
        df_prec = pd.DataFrame(scores_prec, columns=names)
        df_prec.to_csv(directory+"/SMOTEENN_"+name_dataset+"_"+name_antibiotic+"_prec.csv")