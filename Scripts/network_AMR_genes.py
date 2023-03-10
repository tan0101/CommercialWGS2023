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
from matplotlib.collections import LineCollection

import bezier
import networkx as nx
import numpy as np

def curved_edges(G, pos, dist_ratio=0.2, bezier_precision=20, polarity='random'):
    # Get nodes into np array
    edges = np.array(G.edges())
    l = edges.shape[0]

    if polarity == 'random':
        # Random polarity of curve
        rnd = np.where(np.random.randint(2, size=l)==0, -1, 1)
    else:
        # Create a fixed (hashed) polarity column in the case we use fixed polarity
        # This is useful, e.g., for animations
        rnd = np.where(np.mod(np.vectorize(hash)(edges[:,0])+np.vectorize(hash)(edges[:,1]),2)==0,-1,1)
    
    # Coordinates (x,y) of both nodes for each edge
    # e.g., https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    # Note the np.vectorize method doesn't work for all node position dictionaries for some reason
    u, inv = np.unique(edges, return_inverse = True)
    coords = np.array([pos[x] for x in u])[inv].reshape([edges.shape[0], 2, edges.shape[1]])
    coords_node1 = coords[:,0,:]
    coords_node2 = coords[:,1,:]
    
    # Swap node1/node2 allocations to make sure the directionality works correctly
    should_swap = coords_node1[:,0] > coords_node2[:,0]
    coords_node1[should_swap], coords_node2[should_swap] = coords_node2[should_swap], coords_node1[should_swap]
    
    # Distance for control points
    dist = dist_ratio * np.sqrt(np.sum((coords_node1-coords_node2)**2, axis=1))

    # Gradients of line connecting node & perpendicular
    m1 = (coords_node2[:,1]-coords_node1[:,1])/(coords_node2[:,0]-coords_node1[:,0])
    m2 = -1/m1

    # Temporary points along the line which connects two nodes
    # e.g., https://math.stackexchange.com/questions/656500/given-a-point-slope-and-a-distance-along-that-slope-easily-find-a-second-p
    t1 = dist/np.sqrt(1+m1**2)
    v1 = np.array([np.ones(l),m1])
    coords_node1_displace = coords_node1 + (v1*t1).T
    coords_node2_displace = coords_node2 - (v1*t1).T

    # Control points, same distance but along perpendicular line
    # rnd gives the 'polarity' to determine which side of the line the curve should arc
    t2 = dist/np.sqrt(1+m2**2)
    v2 = np.array([np.ones(len(edges)),m2])
    coords_node1_ctrl = coords_node1_displace + (rnd*v2*t2).T
    coords_node2_ctrl = coords_node2_displace + (rnd*v2*t2).T

    # Combine all these four (x,y) columns into a 'node matrix'
    node_matrix = np.array([coords_node1, coords_node1_ctrl, coords_node2_ctrl, coords_node2])

    # Create the Bezier curves and store them in a list
    curveplots = []
    for i in range(l):
        nodes = node_matrix[:,i,:].T
        curveplots.append(bezier.Curve(nodes, degree=3).evaluate_multi(np.linspace(0,1,bezier_precision)).T)
      
    # Return an array of these curves
    curves = np.array(curveplots)
    return curves

mpl.rc('font',family='Arial')

name_dataset = "Ec518" #"SAL143"

folder = "EcoliSNPs" #"SalmonellaSNPs"
results_folder = "SMOTE pre-process" #"Results"
type_data = "combination" #"accessory_core_intergenic"

# Get files in directory:
directory = folder+"/Population Correction/"+results_folder

# Get node information:
genes_info = pd.read_excel(directory+'/'+name_dataset+'MLResults.xlsx', header=[0])
genes_info = genes_info[genes_info['Gene Name'].notna()].reset_index(drop=True)
print(genes_info.shape)
idx = []
for count in range(len(genes_info)):
    if not pd.isnull(genes_info.loc[count,"CARD Gene"]) or not pd.isnull(genes_info.loc[count,"Resfinder Gene"]) or not pd.isnull(genes_info.loc[count,"AMR Finder Gene"]):
        idx.append(count) 

genes_info = genes_info.loc[idx,:].reset_index(drop=True)
print(genes_info.shape)

# Load ARG data:
args_data = pd.read_csv("CARD_ARG_drugclass_NCBI.csv", header=[0])

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

def color_anti(anti_class):
    if anti_class == "beta lactam":
        color_name = "lightcoral"
    elif anti_class == "aminoglycoside":
        color_name = "rosybrown"
    elif anti_class == "phenicol":
        color_name = "lightgreen"
    elif anti_class == "fluoroquinolone":
        color_name = "yellowgreen"
    elif anti_class == "glycopeptide":
        color_name = "cyan"
    elif anti_class == "tetracycline":
        color_name = "violet"
    elif anti_class == "MLSB":
        color_name = "lightskyblue"
    elif anti_class == "trimethoprim & sulfonamide" or anti_class == "trimethoprim" or anti_class == "sulfonamide":
        color_name = "paleturquoise"
    elif anti_class == "MDR":
        color_name = "plum"
    elif anti_class == "peptide":
        color_name = "silver"
    elif anti_class == "mupirocin":
        color_name = "khaki"
    elif anti_class == "fosfomycin":
        color_name = "wheat"
    elif anti_class == "fusidic_acid":
        color_name = "teal"
    elif anti_class == "rifamycin":
        color_name = "mistyrose"
        
    return color_name

def network_chicken(g):
    # chicken
    features_chicken = []
    
    name_anti = []
    for count_anti, anti in enumerate(genes_info["Antibiotic"]):
        #if folder == "EcoliSNPs":
        #    if anti in ["AMC", "CTX-C"]:
        #        continue
        
        id_anti = np.where(genes_info["Antibiotic"] == anti)[0]
        genes_info_anti = genes_info.loc[id_anti,:].reset_index(drop=True)
        features = []
        type_gene = []
        
        for count in range(len(genes_info_anti)):
            if pd.isnull(genes_info_anti.loc[count,"CARD Gene"]):
                if pd.isnull(genes_info_anti.loc[count,"Resfinder Gene"]):
                    if pd.isnull(genes_info_anti.loc[count,"AMR Finder Gene"]):
                        continue
                    else:
                        features.append(genes_info_anti.loc[count,"AMR Finder Gene"].split(" ")[0])
                        type_gene.append(genes_info_anti.loc[count,"Feature"][0])
                else:
                    features.append(genes_info_anti.loc[count,"Resfinder Gene"].split(" ")[0])
                    type_gene.append(genes_info_anti.loc[count,"Feature"][0])    
            else:
                features.append(genes_info_anti.loc[count,"CARD Gene"].split(" ")[0])
                type_gene.append(genes_info_anti.loc[count,"Feature"][0])
            
        name_anti.append(anti)
        
        if len(features) > 0:
            g.add_node(anti, color_="cornflowerblue")
            for count_f, f in enumerate(features):
                id_arg = np.where(args_data["Source"] == f)[0]
                if len(id_arg) == 0:
                    print(f)
                    input("cont")
                f_class = args_data.iloc[id_arg[0],1]
                
                if f not in g.nodes():
                    g.add_node(f, color_=color_anti(f_class), type_ = type_gene[count_f])
                
                g.add_edge(anti,f, color="black")
    
    return g, features_chicken, name_anti

def draw_network(g, axn, labels, name_antibiotic, connect=False):
    node_max_size = 400
    fontsize=8
    node_min_size = 190
        
    node_degree_dict=nx.degree(g)
    nodes_sel = [x for x in g.nodes() if node_degree_dict[x]>0]
    
    color_map=nx.get_node_attributes(g, 'color_')
    type_map=nx.get_node_attributes(g, 'type_')
    
    df_node = pd.DataFrame(columns=["Count","Antibiotics","Class","Type"])
    for n in nodes_sel:
        if n in name_antibiotic:
            continue
        
        neigh = g.neighbors(n)
        neigh_list = []
        for nn in neigh:
            neigh_list.append(nn)
        
        df_node.loc[n,"Count"] = len(neigh_list)    
        df_node.loc[n,"Antibiotics"] = ', '.join(neigh_list) 
        id_arg = np.where(args_data["Source"] == n)[0]
        df_node.loc[n,"Class"] = args_data.iloc[id_arg[0],1]
        
        if type_map[n] == "g":    
            df_node.loc[n,"Type"] = "accessory gene"
        elif type_map[n] == "c":
            df_node.loc[n,"Type"] = "core genome snp"
        elif type_map[n] == "i":
            df_node.loc[n,"Type"] = "intergenic region snp"
                
    df_node.to_csv(directory+'/Node_analysis_genes.csv', index_label="Feature")
        
    #Check distances between nodes for number of iterations
    df = pd.DataFrame(index=g.nodes(), columns=g.nodes())
    for row, data in nx.shortest_path_length(g):
        for col, dist in data.items():
            df.loc[row,col] = dist+1

    df = df.fillna(df.max().max())

    print("Start pos")
    pos = nx.kamada_kawai_layout(g, dist=df.to_dict())
    #pos = nx.kamada_kawai_layout(g)    
    #pos = nx.spring_layout(g,scale=10)
    print("End pos")
    
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
            color_s[i] = "w"
            node_size.append(node_max_size)
            linewidth_val.append(3)
            alpha_val.append(1)
            node_shape_list.append("o") 
        else:
            edge_colors.append(color_n)
            node_size.append(node_min_size)
            alpha_val.append(0.7)  
            linewidth_val.append(1)
            node_shape_list.append("o") 
            
    node_shape_list = np.array(node_shape_list)
    edge_colors = np.array(edge_colors)
    node_size = np.array(node_size)
    alpha_val = np.array(alpha_val)
    linewidth_val = np.array(linewidth_val)
    color_s = np.array(color_s)
    nodes = np.array(nodes_name)
    
    options = {"edgecolors": list(edge_colors), "node_size": list(node_size), "alpha": list(alpha_val), "linewidths":list(linewidth_val)} #[v * 1000 for v in d.values()]
    
    # Produce the curves
    curves = curved_edges(g, pos)
    lc = LineCollection(curves, color='k', alpha=0.3, linewidths=0.5)

    nx.draw_networkx_edges(g, pos, alpha=0, edge_color = colors, width=0.5, ax=axn)
    plt.gca().add_collection(lc)
    nx.draw_networkx_nodes(g, pos, nodelist= nodes, node_shape = "o", node_color=list(color_s), **options, ax=axn)
    nx.draw_networkx_labels(g, pos, labels, font_size=fontsize, font_color="k", ax=axn)
    
    axn.margins(x=0.15)

    if connect == True:
        connectivity = np.round(average_node_connectivity(g),3)
        axn.set_title("Connectivity = {}".format(connectivity), fontsize = 30)

# Plot network
h=nx.Graph()
h, _, name_antibiotic = network_chicken(h)

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
        if n == "determinant_of_bleomycin_resistance":
            legend_node_names.append("bleO")
        else:
            legend_node_names.append(n)
        labels[n] = str(k)
        k+=1        

legend_node_names = np.array(legend_node_names)
legend_node_number = np.array(legend_node_number)

# Networkx        
fig = plt.figure(figsize=(10, 8))

ax0 = fig.add_subplot(111)

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
    if  i == 'cornflowerblue':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='cornflowerblue', label='Antibiotic model',
                        color = 'w', markerfacecolor = 'cornflowerblue', markersize=10, alpha=0.7))
    if i == 'lightcoral':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='lightcoral', label='beta lactam',
                          color = 'w', markerfacecolor = 'lightcoral', markersize=10, alpha=0.7))
    elif i == 'rosybrown':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='rosybrown', label='aminoglycoside',
                          color = 'w', markerfacecolor = 'rosybrown', markersize=10, alpha=0.7))
    elif i == 'lightgreen':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='lightgreen', label='amphenicol',
                          color = 'w', markerfacecolor = 'lightgreen', markersize=10, alpha=0.7))
    elif i == 'yellowgreen':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='yellowgreen', label='fluoroquinolone',
                          color = 'w', markerfacecolor = 'yellowgreen', markersize=10, alpha=0.7))
    elif i == 'cyan':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='cyan', label='glycopeptide',
                          color = 'w', markerfacecolor = 'cyan', markersize=10, alpha=0.7))
    elif i == 'violet':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='violet', label='tetracycline',
                          color = 'w', markerfacecolor = 'violet', markersize=10, alpha=0.7))
    elif i == 'lightskyblue':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='lightskyblue', label='MLSB',
                          color = 'w', markerfacecolor = 'lightskyblue', markersize=10, alpha=0.7))
    elif i == 'paleturquoise':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='paleturquoise', label='trimethoprim & sulfonamide',
                          color = 'w', markerfacecolor = 'paleturquoise', markersize=10, alpha=0.7))
    elif i == 'plum':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='plum', label='MDR',
                          color = 'w', markerfacecolor = 'plum', markersize=10, alpha=0.7))
    elif i == 'silver':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='silver', label='peptide',
                          color = 'w', markerfacecolor = 'silver', markersize=10, alpha=0.7))
    elif i == 'khaki':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='khaki', label='mupirocin',
                          color = 'w', markerfacecolor = 'khaki', markersize=10, alpha=0.7))
    elif i == 'teal':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='teal', label='fusidic_acid',
                          color = 'w', markerfacecolor = 'teal', markersize=10, alpha=0.7))
    elif i == 'mistyrose':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='mistyrose', label='rifamycin',
                          color = 'w', markerfacecolor = 'mistyrose', markersize=10, alpha=0.7))
    elif i == 'wheat':
        legend_elements.append(Line2D([], [], marker='o', markeredgecolor='wheat', label='fosfomycin',
                          color = 'w', markerfacecolor = 'wheat', markersize=10, alpha=0.7))
    
ax0.legend(handles = legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.01),
          fancybox=True, shadow=True, ncol=4, fontsize = 12,
          title="Nodes", title_fontsize=15)

plt.tight_layout()


handles = []
for i in range(len(legend_node_names)):
    id_arg = np.where(args_data["Source"] == legend_node_names[i])[0]
    f_class = args_data.iloc[id_arg[0],1]
    handles.append(plt.text(1.2,0.5,str(i+1), transform=plt.gcf().transFigure,
         bbox={"boxstyle" : "circle", "color":color_anti(f_class), "alpha":0.7, "pad":0.1}))

handlermap = {type(handles[0]) : TextHandler()}
leg = plt.figlegend(handles, legend_node_names, handler_map=handlermap, bbox_to_anchor=[1.48, 0.5], loc='center right', ncol=2,
                    fancybox=True, shadow=True, fontsize = 13, title="Genes Legend", title_fontsize=16)

#plt.savefig('Figure6b_C_H.png')

plt.savefig(directory+'/Network_AMR_Genes.svg', dpi=300, bbox_inches='tight')
plt.savefig(directory+'/Network_AMR_Genes.png', dpi=300, bbox_inches='tight')
