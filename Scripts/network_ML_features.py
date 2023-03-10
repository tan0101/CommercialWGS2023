import pandas as pd
import glob
import seaborn as sns
import matplotlib as mpl
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
import copy
from networkx.algorithms.connectivity.connectivity import average_node_connectivity
from matplotlib.lines import Line2D
from itertools import chain
from scipy.stats import ranksums
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from collections import Counter
from pathlib import Path


mpl.rc('font',family='Arial')

class TextHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,xdescent, ydescent,
                        width, height, fontsize,trans):
        h = copy.copy(orig_handle)
        h.set_position((width/2.,height/2.))
        h.set_transform(trans)
        h.set_ha("center");h.set_va("center")
        fp = orig_handle.get_font_properties().copy()
        fp.set_size(fontsize)
        # uncomment the following line, 
        # if legend symbol should have the same size as in the plot
        h.set_font_properties(fp)
        return [h]

def network_chicken(g, type = []):
    # chicken
    features_chicken = []

    if type == "Ecoli":
        name_dataset = "Ec518" #"SAL143"
        folder = "EcoliSNPs"
        results_folder = "SMOTE pre-process" #"Results"
        type_data = "combination" #"accessory_core_intergenic"
    else:
        name_dataset = "SAL143"
        folder = "SalmonellaSNPs"
        results_folder = "Results"
        type_data = "accessory_core_intergenic"
    
    # Load Antibiotic Data:
    antibiotic_df = pd.read_csv(folder+"/"+name_dataset+'_AMR_data_RSI.csv', header = [0])

    # Get files in directory:
    directory = folder+"/Population Correction/"+results_folder+"/"+type_data
    
    print(antibiotic_df.columns[1:])
    name_anti = []
    for count, anti in enumerate(antibiotic_df.columns[1:]):
        if folder == "EcoliSNPs":
            if anti in ["AMC", "CTX-C"]:
                continue
            
        file_name = directory+"/features_"+name_dataset+"_"+anti+'.csv'
        
        my_file = Path(file_name)

        try:
            my_abs_path = my_file.resolve(strict=True)
        except FileNotFoundError:
            continue
        
        data=pd.read_csv(file_name, index_col=[0], header=[0])
        features=data.index 
        name_anti.append(anti)
        
        if len(features) > 0:
            g.add_node(anti, color_="silver")
            for f in features:
                if len(f) == 13:
                    g.add_node(f, color_="darkorange")                    
                else:
                    if f[0] == "c":
                        g.add_node(f, color_="forestgreen")
                    elif f[0] == "i":
                        g.add_node(f, color_="royalblue")
                    else:
                        g.add_node(f, color_="orange")
                        
                g.add_edge(anti,f, color="black")
    
    return g, features_chicken, name_anti

def draw_network(g, axn, labels, name_antibiotic, connect=False):
    node_max_size = 240
    fontsize=6
    node_min_size = 3
        
    node_degree_dict=nx.degree(g)
    nodes_sel = [x for x in g.nodes() if node_degree_dict[x]>0]
    
    color_map=nx.get_node_attributes(g, 'color_')
    
    df_node = pd.DataFrame(columns=["Count","Antibiotics"])
    for n in nodes_sel:
        if n in name_antibiotic:
            continue
        
        neigh = g.neighbors(n)
        neigh_list = []
        for nn in neigh:
            neigh_list.append(nn)
        
        c = color_map.get(n)        
        if c == "darkorange":
            df_node.loc[n,"Type"] = "13-mer"
        elif c == "forestgreen":
            df_node.loc[n,"Type"] = "core genome snp"
        elif c == "royalblue":
            df_node.loc[n,"Type"] = "intergenic region snp"
        elif c == "orange":        
            df_node.loc[n,"Type"] = "accessory gene" 
            
        df_node.loc[n,"Count"] = len(neigh_list)    
        df_node.loc[n,"Antibiotics"] = ', '.join(neigh_list)  
            
    pos = nx.spring_layout(g,scale=3)
    
    color_s=[color_map.get(x) for x in g.nodes()]
    edges = g.edges()
    colors = [g[u][v]['color'] for u,v in edges]
    node_size = []
    edge_colors = []
    linewidth_val = []
    alpha_val = []
    node_shape_list = []
    nodes_name = []
    for i, n in enumerate(g.nodes):
        color_n = color_map.get(n)
        nodes_name.append(n)
        if n in name_antibiotic:
            edge_colors.append(color_s[i])
            color_s[i] = "white"
            node_size.append(node_max_size)
            linewidth_val.append(3)
            alpha_val.append(1)
            node_shape_list.append("o") 
        else:
            edge_colors.append(color_n)
            node_size.append(node_min_size)
            
            alpha_val.append(1)  
            linewidth_val.append(1) 
            node_shape_list.append("o") 
            
    node_shape_list = np.array(node_shape_list)
    edge_colors = np.array(edge_colors)
    node_size = np.array(node_size)
    alpha_val = np.array(alpha_val)
    linewidth_val = np.array(linewidth_val)
    color_s = np.array(color_s)
    nodes = np.array(nodes_name)
    
    id_o = np.where(node_shape_list == "o")[0]
    options_o = {"edgecolors": list(edge_colors[id_o]), "node_size": list(node_size[id_o]), "alpha": list(alpha_val[id_o]), "linewidths":list(linewidth_val[id_o])} #[v * 1000 for v in d.values()]
    
    nx.draw_networkx_nodes(g, pos, nodelist= nodes[id_o], node_shape = "o", node_color=list(color_s[id_o]), **options_o, ax=axn)
    
    nx.draw_networkx_edges(g, pos, alpha=0.2, edge_color = colors, width=0.2, ax=axn)
    nx.draw_networkx_labels(g, pos, labels, font_size=fontsize, font_color="k", ax=axn)
    axn.margins(x=0.15)

    if connect == True:
        connectivity = np.round(average_node_connectivity(g),3)
        axn.set_title("Connectivity = {}".format(connectivity), fontsize = 30)

# Plot Ecoli network
h=nx.Graph()
h, _, name_antibiotic = network_chicken(h, type="Ecoli")

color_map=nx.get_node_attributes(h, 'color_')
legend_node_names = []
legend_node_number = []
labels = {}
k = 1
for n in h.nodes:
    color_n = color_map.get(n)
    if n in name_antibiotic:
        labels[n] = n
    else:    
        legend_node_number.append(str(k))
        legend_node_names.append(n)
        labels[n] = ""
        k+=1        

legend_node_names = np.array(legend_node_names)
legend_node_number = np.array(legend_node_number)

# Networkx        
fig = plt.figure(figsize=(10, 15))

ax0 = fig.add_subplot(211)

plt.rcParams.update({'font.size': 20})
draw_network(h,ax0,labels, name_antibiotic)

color_map=nx.get_node_attributes(h, 'color_')

c_map = []
for key in color_map.keys():
    c_map.append(color_map[key])

c_map = Counter(c_map)
print(c_map)
input("cont")

legend_elements = []

for i in c_map.keys():
    if  i == 'silver':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='silver', label='Antibiotic',
                          color = 'w', markerfacecolor = 'silver', markersize=10, alpha=1))
    elif i == 'darkorange':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='darkorange', label='13-mers',
                          color = 'w', markerfacecolor = 'darkorange', markersize=10, alpha=1))
    elif i == 'forestgreen':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='forestgreen', label='Core Genome SNPs',
                          color = 'w', markerfacecolor = 'forestgreen', markersize=10, alpha=1))
    elif i == 'royalblue':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='royalblue', label='Intergenic Region SNPs',
                          color = 'w', markerfacecolor = 'royalblue', markersize=10, alpha=1))
    elif i == 'orange':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='orange', label='Accessory Genes',
                          color = 'w', markerfacecolor = 'orange', markersize=10, alpha=1))
    
ax0.legend(handles = legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.01),
          fancybox=True, shadow=True, ncol=4, fontsize = 12,
          title="Nodes", title_fontsize=15)
          

# Plot Salmonella network
h=nx.Graph()
h, _, name_antibiotic = network_chicken(h, type="Salmonella")

color_map=nx.get_node_attributes(h, 'color_')
legend_node_names = []
legend_node_number = []
labels = {}
k = 1
for n in h.nodes:
    color_n = color_map.get(n)
    if n in name_antibiotic:
        labels[n] = n
    else:    
        legend_node_number.append(str(k))
        legend_node_names.append(n)
        labels[n] = ""
        k+=1        

legend_node_names = np.array(legend_node_names)
legend_node_number = np.array(legend_node_number)

# Networkx        
ax1 = fig.add_subplot(212)

draw_network(h,ax1,labels, name_antibiotic)

color_map=nx.get_node_attributes(h, 'color_')

plt.tight_layout()
plt.savefig('Figure_S9.svg', dpi=300, bbox_inches='tight')
