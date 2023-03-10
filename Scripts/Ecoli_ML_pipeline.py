# -*- coding: utf-8 -*-
"""
Created on Fri Sep  24 20:35:00 2021

@author: Alexandre Maciel Guerra
"""

import numpy as np
import pandas as pd
import sys
import os
import pickle

from collections import Counter
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, chi2
from sklearn.model_selection import StratifiedKFold, GridSearchCV,cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer, cohen_kappa_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from pathlib import Path

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

if __name__ == "__main__":
    name_dataset = "Ec518" 
    folder = "EcoliSNPs"
    results_folder = "combination"
    type_data = "combination" #intergenic #accessory #core_genome
    res_folder = "SMOTE pre-process"
    
    # Load Data Core:
    data_core_df = pd.read_csv(folder+"/"+name_dataset+'_core_genome_snps_data.csv', header = [0], index_col=[0])
    data_core_df = data_core_df.T
    data_core_df.sort_index(key=lambda x: x.str.lower(), inplace=True)
    sample_name_core = np.array(data_core_df.index)
    
    # Load Metadata:
    metadata_df = pd.read_csv(folder+"/"+name_dataset+'_metadata.csv', header = [0], index_col=[0])
    metadata_df.sort_index(key=lambda x: x.str.lower(),inplace=True)
    
    id_not_human = []
    for count, s in enumerate(metadata_df["Sample Host"]):
        if "Human" not in s:
            id_not_human.append(count)
            
    # Load Antibiotic Data:
    antibiotic_df = pd.read_csv(folder+"/"+name_dataset+'_AMR_data_RSI.csv', header = [0])
    samples_AMR = np.array(antibiotic_df[antibiotic_df.columns[0]])
       
    print(np.array_equal(sample_name_core, samples_AMR))
    
    # Change order of samples in the AMR dataframes
    order_id = []
    for count, s_name in enumerate(sample_name_core):
        idx = np.where(samples_AMR == s_name)[0]
        order_id.append(idx[0])
        
    antibiotic_df = antibiotic_df.iloc[order_id, :].reset_index()
    antibiotic_df.drop(columns="index",axis=1,inplace=True)
    samples_AMR = np.array(antibiotic_df[antibiotic_df.columns[0]])
    
    print("all samples length = {}".format(len(samples_AMR)))
    
    print(np.array_equal(sample_name_core, samples_AMR))
    
    if len(id_not_human) > 0:
        antibiotic_df = antibiotic_df.iloc[id_not_human, :].reset_index()
        antibiotic_df.drop(columns="index",axis=1,inplace=True)
        samples_AMR = np.array(antibiotic_df[antibiotic_df.columns[0]])
    
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
            samples_sel = np.delete(samples_AMR,idx_NaN)
        else:
            samples_sel = samples_AMR
        
        # Check minimum number of samples:
        count_class = Counter(target)
        print(count_class)
        
        if len(count_class) < 2:
            continue

        most_common = count_class.most_common(1)[0]

        if count_class[0] < 15 or count_class[1] < 15:
            continue 
        
        if count_class[0] > count_class[1] and count_class[1] < 0.10*count_class[0]:
            continue
        
        if count_class[1] > count_class[0] and count_class[0] < 0.10*count_class[1]:
            continue 
        
        if results_folder == "combination":
            data_acc = []
            data_core = []
            data_inter = []
            
            df_features_acc = pd.DataFrame()
            df_features_core = pd.DataFrame()
            df_features_inter = pd.DataFrame()
            
            file_name = folder+"/Population Correction/accessory_genes/Chi Square Features/data_"+name_dataset+"_"+name_antibiotic+'.pickle'
            my_file = Path(file_name)

            try:
                my_abs_path = my_file.resolve(strict=True)
            except FileNotFoundError:
                continue
            else:
                with open(file_name, 'rb') as f:
                    data_acc = pickle.load(f)
                df_features_acc = pd.read_csv(folder+"/Population Correction/accessory_genes/Chi Square Features/"+name_dataset+"_"+name_antibiotic+"_model_pvalue.csv", header=[0], index_col=[0])
            
            
            file_name = folder+"/Population Correction/core_genome_snps/Chi Square Features/data_"+name_dataset+"_"+name_antibiotic+'.pickle'
            my_file = Path(file_name)

            try:
                my_abs_path = my_file.resolve(strict=True)
            except FileNotFoundError:
                continue
            else:
                with open(file_name, 'rb') as f:
                    data_core = pickle.load(f)
                df_features_core = pd.read_csv(folder+"/Population Correction/core_genome_snps/Chi Square Features/"+name_dataset+"_"+name_antibiotic+"_model_pvalue.csv", header=[0], index_col=[0])
            
            
            file_name = folder+"/Population Correction/intergenic_snps/Chi Square Features/data_"+name_dataset+"_"+name_antibiotic+'.pickle'
            my_file = Path(file_name)

            try:
                my_abs_path = my_file.resolve(strict=True)
            except FileNotFoundError:
                continue
            else:
                with open(file_name, 'rb') as f:
                    data_inter = pickle.load(f)
                df_features_inter = pd.read_csv(folder+"/Population Correction/intergenic_snps/Chi Square Features/"+name_dataset+"_"+name_antibiotic+"_model_pvalue.csv", header=[0], index_col=[0])
            
            data = np.concatenate((data_acc, data_core, data_inter), axis=1)
            df_features = pd.concat([df_features_acc, df_features_core, df_features_inter], axis=0)
            features_anti = df_features.index
        else:
            file_name = folder+"/Population Correction/"+results_folder+"/Chi Square Features/data_"+name_dataset+"_"+name_antibiotic+'.pickle'
            my_file = Path(file_name)

            try:
                my_abs_path = my_file.resolve(strict=True)
            except FileNotFoundError:
                continue
            else:
                with open(file_name, 'rb') as f:
                    data = pickle.load(f)
            
            df_features = pd.read_csv(folder+"/Population Correction/"+results_folder+"/Chi Square Features/"+name_dataset+"_"+name_antibiotic+"_model_pvalue.csv", header=[0], index_col=[0])
            features_anti = df_features.index
        
        print(data.shape)
        print(df_features.shape)
        
        data_nonSMOTE = data
        
        sm = SMOTE(random_state=42)
        data, target = sm.fit_resample(data, target)
        print('Resampled dataset shape %s' % Counter(target))
        
        # Preprocessing - Feature Selection
        std_scaler = MinMaxScaler()
        data = std_scaler.fit_transform(data)
        
        _, pvalue = chi2(data, target)
        
        threshold = 1E-5

        #threshold = 0.001/len(features_anti)
        ind = np.where(pvalue < threshold)[0]
        
        if len(ind) == 0:
            continue
            
        features_anti = features_anti[ind]
        n_features = len(features_anti)
        data = data[:, ind]
        
        print("After select from model:{}".format(data.shape))
        
        selector1 = SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=50, random_state=0)).fit(data, target)   
        coef = selector1.estimator_.feature_importances_
        cols_model = selector1.get_support(indices=True)
        
        data = data[:, cols_model]
        print("Optimal number of features : %d" % data.shape[1])
        
        concat_array = np.zeros((len(cols_model),2))
        concat_array[:,0] = pvalue[ind[cols_model]]
        concat_array[:,1] = coef[cols_model]
        
        features_anti = features_anti[cols_model]
        
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
            inner_cv = StratifiedKFold(n_splits=inner_loop_cv, shuffle=True, random_state=i)
            outer_cv = StratifiedKFold(n_splits=outer_loop_cv, shuffle=True, random_state=i)
        
            k = 0
        
            for name, clf in zip(names, classifiers):
                model = Pipeline([('clf', clf)]) 
                
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

        
        directory = folder+"/Population Correction/"+res_folder+"/"+type_data
        if not os.path.exists(directory):
            os.makedirs(directory)

        results_df=pd.DataFrame(results, columns=names, index=names_scr)
        results_df.to_csv(directory+"/SMOTE_results_"+name_dataset+"_"+name_antibiotic+".csv")

        df_auc = pd.DataFrame(scores_auc, columns=names)
        df_auc.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_auc.csv")
        
        df_acc = pd.DataFrame(scores_acc, columns=names)
        df_acc.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_acc.csv")
        
        df_sens = pd.DataFrame(scores_sens, columns=names)
        df_sens.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_sens.csv")
        
        df_spec = pd.DataFrame(scores_spec, columns=names)
        df_spec.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_spec.csv")
        
        df_kappa = pd.DataFrame(scores_kappa, columns=names)
        df_kappa.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_kappa.csv")
        
        df_prec = pd.DataFrame(scores_prec, columns=names)
        df_prec.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_prec.csv")
        
        df_features = pd.DataFrame(concat_array, columns = ["pvalue", "coef"], index=features_anti)
        df_features.to_csv(directory+"/features_"+name_dataset+"_"+name_antibiotic+".csv")
            
        with open(directory+"/data_"+name_dataset+"_"+name_antibiotic+'.pickle', 'wb') as f:
            pickle.dump(data_nonSMOTE[:,ind[cols_model]], f)
        