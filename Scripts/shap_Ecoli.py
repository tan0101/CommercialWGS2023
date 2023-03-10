import pandas as pd
import numpy as np
import shap

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import tree

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from collections import Counter
import pickle
import sys
import os

from sklearn.model_selection import train_test_split



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

    plt.figure(figsize=(15,20))
     
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

    k = 1
    
    for name_antibiotic in np.sort(antibiotic_df.columns[1:]):
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
            file_name = folder+"/Population Correction/"+res_folder+"/"+type_data+"/data_"+name_dataset+"_"+name_antibiotic+'.pickle'
            my_file = Path(file_name)

            try:
                my_abs_path = my_file.resolve(strict=True)
            except FileNotFoundError:
                continue
            else:
                with open(file_name, 'rb') as f:
                    data = pickle.load(f)
                df_features = pd.read_csv(folder+"/Population Correction/"+res_folder+"/"+type_data+"/features_"+name_dataset+"_"+name_antibiotic+".csv", header=[0], index_col=[0])
            
        df_genes = pd.read_excel(folder+"/"+name_dataset+"_MLFeaturesSummary.xlsx", sheet_name="MLFeatures")
        
        df_genes = df_genes.loc[np.where(df_genes["Antibiotic"] == name_antibiotic)[0], :]
        df_genes.reset_index(inplace=True, drop=True)
        
        features_name = []
        for name in df_features.index:
            idx = np.where(df_genes["Feature"] == name)[0]
            if len(idx)>0:
                gene = np.array(df_genes.loc[idx,"Gene Name"])
                features_name.append(gene[0])
            else:
                print(name)
                features_name.append("")
    
        #X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size = 0.2)

        model = ExtraTreesClassifier(n_estimators=50)
        # Fit the Model
        model.fit(data, target)

        # load JS visualization code to notebook
        shap.initjs()

        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
        explainer = shap.TreeExplainer(model, data)
        shap_values = np.array(explainer.shap_values(data))

        print(shap_values.shape)

        ax = plt.subplot(7,3,k) #ax = plt.subplot(4,3,k)
        shap.summary_plot(shap_values[1],data, max_display=10, feature_names=np.array(features_name),show=False, plot_size=None)
        ax.set_title(name_antibiotic)
        k+=1
    
    # Get files in directory:
    directory = folder+"/Population Correction/"+res_folder+"/"+type_data+"/Shap Summary Plot"

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(directory+'/Shap_'+name_dataset+'.svg', dpi=300, bbox_inches='tight')      

                
        