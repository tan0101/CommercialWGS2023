# libraries
from networkx.algorithms.graphical import is_valid_degree_sequence_havel_hakimi
from networkx.generators.joint_degree_seq import is_valid_joint_degree
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

import colorcet as cc
import seaborn as sns
from pathlib import Path

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import dice,jaccard, squareform,pdist,cdist, hamming, euclidean, cosine, braycurtis, chebyshev, canberra, minkowski, sqeuclidean
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import Counter
from matplotlib.lines import Line2D
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
from scipy.spatial import distance
from scipy.cluster import hierarchy
import copy
from matplotlib.legend_handler import HandlerBase

from community import community_louvain

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
    mpl.rc('font',family='Arial')
    
    for specie in ["Ecoli", "Salmonella"]:
        if specie == "Ecoli":
            name_dataset = "EC518" 
            folder = "EcoliSNPs"
            
            # Load Data Core:
            data_core_df = pd.read_csv(folder+"/"+name_dataset+'_core_genome_snps_data.csv', header = [0], index_col=[0])
            data_core_df = data_core_df.T
            data_core_df.sort_index(key=lambda x: x.str.lower(), inplace=True)
            sample_name_core = np.array(data_core_df.index)
            
            # Load Metadata:
            metadata_df = pd.read_csv(folder+"/"+name_dataset+'_metadata.csv', header = [0], index_col=[0])
            metadata_df = metadata_df.loc[sample_name_core,:]      

            print(np.array_equal(sample_name_core,metadata_df.index))    
            
            samples_name = np.array(data_core_df.index)
            snp_matrix = np.array(data_core_df)
            snp_matrix[snp_matrix>0] = 1

            print(np.array_equal(sample_name_core,metadata_df.index))

            adj_matrix = pairwise_distances(snp_matrix,metric='hamming')
            adj_matrix[np.tril_indices(len(samples_name), 0)] = np.nan
            distribution_adj = adj_matrix.flatten()
            mask = np.where(~np.isnan(distribution_adj))[0]
            distribution_adj = data_core_df.shape[1]*distribution_adj[mask]
            print(np.mean(distribution_adj))
            print(np.amax(distribution_adj))
            print(np.median(distribution_adj))

            # First quartile (Q1) 
            Q1 = np.percentile(distribution_adj, 25, method = 'midpoint') 
            print(Q1)
            
            # Third quartile (Q3) 
            Q3 = np.percentile(distribution_adj, 75, method = 'midpoint') 
            print(Q3)
            
            # Interquaritle range (IQR) 
            IQR = Q3 - Q1 
            
            print(IQR) 
            print(distribution_adj.shape)

            #n, bins, patches = plt.hist(distribution_adj)

            adj_matrix = adj_matrix*data_core_df.shape[1]
            adj_matrix[adj_matrix>15] = np.nan

            
            print(np.nanmax(adj_matrix))
            adj_df = pd.DataFrame(data=adj_matrix,index=samples_name, columns=samples_name)
            lst = adj_df.stack().reset_index()
            lst = lst.rename(columns={lst.columns[0]:"from", lst.columns[1]:"to", lst.columns[2]:"edge_value"})

            lst.to_csv("Ecoli_edges_network.csv")
            print(adj_matrix.shape)
            print(lst.shape)

            T=nx.from_pandas_edgelist(df=lst, source='from', target='to', edge_attr='edge_value', create_using=nx.Graph() )

            # compute the best partition
            partition = community_louvain.best_partition(T)
            partition_df = pd.DataFrame()

            for key in partition.keys():
                partition_df.loc[key,"Community"] = partition[key]

            partition_df.to_csv("Ecoli_communities.csv")
            
            carac = metadata_df
            sample_carac = carac.index

            carac = carac.reindex(T.nodes())        
            
            # And I need to transform my categorical column in a numerical value: group1->1, group2->2...
            carac['Sample Host']=pd.Categorical(carac['Sample Host'])
            my_color_node_Type = carac['Sample Host'].cat.codes

            carac['Time']=pd.Categorical(carac['Time'])
            my_color_node_Date = carac['Time'].cat.codes

            carac['Sample Source']=pd.Categorical(carac['Sample Source'])
            my_color_node_Source = carac['Sample Source'].cat.codes

            carac['Phylogroup']=pd.Categorical(carac['Phylogroup'])
            my_color_node_Source = carac['Phylogroup'].cat.codes

            lst['edge_value'] = pd.Categorical(lst['edge_value'])
            my_color_edge = lst['edge_value'].cat.codes
                    
            ColorLegend_Node_Type = {'Chicken': 0,'Environment': 1}
            ColorLegend_Node_Date = {'T1': 0, 'T2': 1, 'T3': 2}
            ColorLegend_Node_Source = {'Chicken faeces': 0,'Chicken carcass': 1,'Chicken cecal': 2, 'Feed': 3,
                'Soil': 4, 'Sewage': 5, 'Chicken feather': 6,'Abattoir environment': 7,'Barn environment': 8,
                'Chicken anal': 9, 'Water': 10}
            ColorLegend_Node_Phylogroup = {'A': 0,'B1': 1,'C': 2, 'D': 3,
                'E': 4, 'F': 5, 'G': 6,'U': 7,'U/cryptic': 8,
                'cladeI': 9}
            ColorLegend_Edge = {}
            Legend_Edge = {}
            #uni_edge_val = np.unique(lst["edge_value"])

            edge_value_array = []
            for edge in T.edges():
                edge_val = T.get_edge_data(edge[0], edge[1])
                edge_value_array.append(np.ceil(edge_val["edge_value"]))
            uni_edge_val = np.unique(edge_value_array)

            for count, n_gene in enumerate(uni_edge_val):
                ColorLegend_Edge[n_gene] = count
                Legend_Edge[count] = n_gene

            
            values_edge = []
            for edge in T.edges():
                edge_val = T.get_edge_data(edge[0], edge[1])
                values_edge.append(ColorLegend_Edge[np.ceil(edge_val['edge_value'])])

            values_node_Type = []
            values_node_Date = []
            values_node_Source = []
            values_node_Phylogroup = []
            for node  in T.nodes():
                values_node_Type.append(ColorLegend_Node_Type[carac.loc[node,'Sample Host']])
                values_node_Date.append(ColorLegend_Node_Date[carac.loc[node,'Time']])
                values_node_Source.append(ColorLegend_Node_Source[carac.loc[node,'Sample Source']])
                values_node_Phylogroup.append(ColorLegend_Node_Phylogroup[carac.loc[node,'Phylogroup']])

            values_node_Type = np.array(values_node_Type)
            values_node_Date = np.array(values_node_Date)
            values_node_Source = np.array(values_node_Source)
            values_node_Phylogroup = np.array(values_node_Phylogroup)
            
            print(len(T.nodes()))
            id_node_env = np.where(values_node_Type == 1)[0]
            env_len = len(id_node_env)
            print("env = {}".format(env_len))
            id_node_chicken = np.where(values_node_Type == 0)[0]
            chicken_len = len(id_node_chicken)
            print("chicken = {}".format(chicken_len))
            
            # compute maximum value s.t. all colors can be normalised
            maxval_node_Type = np.max(values_node_Type) 
            maxval_node_Date = np.max(values_node_Date) 
            maxval_node_Source = np.max(values_node_Source) 
            maxval_node_Phylogroup = np.max(values_node_Phylogroup) 
            maxval_edge = np.max(values_edge) 
            
            # get colormap
            cmap_Type=cm.Paired
            cmap_Date=cm.Set1
            cmap_Phylogroup=cm.tab10
            cmap_Source=cm.Paired

            if len(np.unique(values_edge)) < 21:
                cmap_edge=cm.tab20 #Accent
            else:
                cmap_edge=cm.gist_rainbow

            pos = nx.nx_agraph.graphviz_layout(T)

            nodes = np.array(T.nodes())
            fig, ax = plt.subplots(nrows = 2, ncols=3, figsize=(15,10))
            ax = ax.ravel()
    
            ## Nodes Time
            nx.draw_networkx_nodes(T, pos, node_size=15, nodelist= nodes[id_node_chicken], node_color=[cmap_Date(v/maxval_node_Date) for v in values_node_Date[id_node_chicken]], 
                node_shape = 'o', ax=ax[0])
            nx.draw_networkx_nodes(T, pos, node_size=15, nodelist= nodes[id_node_env], node_color=[cmap_Date(v/maxval_node_Date) for v in values_node_Date[id_node_env]], 
                node_shape = '*', ax=ax[0])
            
            nx.draw_networkx_edges(T,pos,alpha = 0.6, edge_color=[cmap_edge(v/maxval_edge) for v in values_edge], edge_cmap=cmap_edge, ax=ax[0], width=0.6)
            legend_elements = []
            for v in set(values_node_Date):
                if v == 0:
                    label = "T1"
                elif v == 1:
                    label = "T2"
                else:
                    label = "T3"
                legend_elements.append(Line2D([], [], marker='o', markeredgecolor=cmap_Date(v/maxval_node_Date), label=label,
                                color = 'w', markerfacecolor = cmap_Date(v/maxval_node_Date), markersize=10))
            
            ax[0].legend(handles = legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.03),
                fancybox=True, shadow=True, ncol=5, fontsize=9)

            ## Nodes Source Type

            nx.draw_networkx_nodes(T, pos, node_size=15, nodelist= nodes[id_node_chicken], node_color=[cmap_Source(v/maxval_node_Source) for v in values_node_Source[id_node_chicken]], 
                node_shape = 'o', ax=ax[1])
            nx.draw_networkx_nodes(T, pos, node_size=15, nodelist= nodes[id_node_env], node_color=[cmap_Source(v/maxval_node_Source) for v in values_node_Source[id_node_env]], 
                node_shape = '*', ax=ax[1])
            
            nx.draw_networkx_edges(T,pos,alpha = 0.6, edge_color=[cmap_edge(v/maxval_edge) for v in values_edge], edge_cmap=cmap_edge, ax=ax[1], width=0.6)
            legend_elements = []
            for v in set(values_node_Source):
                if v == 0:
                    label = "Chicken faeces"
                elif v == 1:
                    label = "Chicken carcass"
                elif v == 2:
                    label = "Chicken cecal"
                elif v == 3:
                    label = "Feed"
                elif v == 4:
                    label = "Soil"
                elif v == 5:
                    label = "Sewage"
                elif v == 6:
                    label = "Chicken feather"
                elif v == 7:
                    label = "Abattoir environment"
                elif v == 8:
                    label = "Barn environment"
                elif v == 9:
                    label = "Chicken anal"
                else:
                    label = "Water"
                legend_elements.append(Line2D([], [], marker='o', markeredgecolor=cmap_Source(v/maxval_node_Source), label=label,
                                color = 'w', markerfacecolor = cmap_Source(v/maxval_node_Source), markersize=10))
            
            ax[1].legend(handles = legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.03),
                fancybox=True, shadow=True, ncol=3, fontsize=9)
            ax[1].set_title("Escherichia coli", style='italic')

            ## Nodes Phylogroup

            nx.draw_networkx_nodes(T, pos, node_size=15, nodelist= nodes[id_node_chicken], node_color=[cmap_Phylogroup(v/maxval_node_Phylogroup) for v in values_node_Phylogroup[id_node_chicken]], 
                node_shape = 'o', ax=ax[2])
            nx.draw_networkx_nodes(T, pos, node_size=15, nodelist= nodes[id_node_env], node_color=[cmap_Phylogroup(v/maxval_node_Phylogroup) for v in values_node_Phylogroup[id_node_env]], 
                node_shape = '*', ax=ax[2])
            
            nx.draw_networkx_edges(T,pos,alpha = 0.6, edge_color=[cmap_edge(v/maxval_edge) for v in values_edge], edge_cmap=cmap_edge, ax=ax[2], width=0.6)
            legend_elements = []
            for v in set(values_node_Phylogroup):
                if v == 0:
                    label = "A"
                elif v == 1:
                    label = "B1"
                elif v == 2:
                    label = "C"
                elif v == 3:
                    label = "D"
                elif v == 4:
                    label = "E"
                elif v == 5:
                    label = "F"
                elif v == 6:
                    label = "G"
                elif v == 7:
                    label = "U"
                elif v == 8:
                    label = "U/cryptic"
                elif v == 9:
                    label = "cladeI"

                legend_elements.append(Line2D([], [], marker='o', markeredgecolor=cmap_Phylogroup(v/maxval_node_Phylogroup), label=label,
                                color = 'w', markerfacecolor = cmap_Phylogroup(v/maxval_node_Phylogroup), markersize=10))
            
            ax[2].legend(handles = legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.03),
                fancybox=True, shadow=True, ncol=4, fontsize=9)

            ## Legend Edges

            handles = []
            legend_node_names = ["Chicken", "Environment"]
            handles.append(Line2D([], [], marker='o', label="Chicken", markersize=10, markeredgecolor="grey",
                        color = 'w', markerfacecolor = "grey"))
            handles.append(Line2D([], [], marker='*', label="Environment", markersize=12, markeredgecolor="grey",
                        color = 'w', markerfacecolor = "grey"))
                

            leg = plt.figlegend(handles, legend_node_names, bbox_to_anchor=[1.02, 0.7], loc='center right', ncol=1,
                                fancybox=True, shadow=True, fontsize=9, title="Nodes", title_fontsize=12)

            handles = []
            legend_edge_names = []
            for v in set(values_edge):
                label = np.round(Legend_Edge[v],4)
                handles.append(plt.text(0.95,0.5,"   ", transform=plt.gcf().transFigure,
                bbox={"boxstyle" : "square", "color":cmap_edge(v/maxval_edge), "alpha": 0.6, "pad":0.1}))
                legend_edge_names.append(str(label))

            handlermap = {type(handles[0]) : TextHandler()}
            leg = plt.figlegend(handles, legend_edge_names, handler_map=handlermap, bbox_to_anchor=[1.03, 0.4], loc='center right', ncol=1,
                                fancybox=True, shadow=True, fontsize=9, title="Number of SNPs", title_fontsize=12)

        else:
            name_dataset = "SAL143" 
            folder = "SalmonellaSNPs"
            
            # Load Data Core:
            data_core_df = pd.read_csv(folder+"/"+name_dataset+'_core_genome_snps.csv', header = [0], index_col=[0])
            data_core_df = data_core_df.T
            data_core_df.sort_index(key=lambda x: x.str.lower(), inplace=True)
            sample_name_core = np.array(data_core_df.index)
            
            # Load Metadata:
            metadata_df = pd.read_csv(folder+"/"+name_dataset+'_metadata.csv', header = [0], index_col=[0])
            metadata_df = metadata_df.loc[sample_name_core,:]      

            print(np.array_equal(sample_name_core,metadata_df.index))    
            
            samples_name = np.array(data_core_df.index)
            snp_matrix = np.array(data_core_df)
            snp_matrix[snp_matrix>0] = 1

            print(np.array_equal(sample_name_core,metadata_df.index))

            adj_matrix = pairwise_distances(snp_matrix,metric='hamming')
            adj_matrix[np.tril_indices(len(samples_name), 0)] = np.nan
            distribution_adj = adj_matrix.flatten()
            mask = np.where(~np.isnan(distribution_adj))[0]
            distribution_adj = data_core_df.shape[1]*distribution_adj[mask]
            print(np.mean(distribution_adj))
            print(np.amax(distribution_adj))
            print(np.median(distribution_adj))

            # First quartile (Q1) 
            Q1 = np.percentile(distribution_adj, 25, method = 'midpoint') 
            print(Q1)
            
            # Third quartile (Q3) 
            Q3 = np.percentile(distribution_adj, 75, method = 'midpoint') 
            print(Q3)
            
            # Interquaritle range (IQR) 
            IQR = Q3 - Q1 
            
            print(IQR) 
            print(distribution_adj.shape)

            #n, bins, patches = plt.hist(distribution_adj)

            adj_matrix = adj_matrix*data_core_df.shape[1]
            adj_matrix[adj_matrix>15] = np.nan

            
            print(np.nanmax(adj_matrix))
            adj_df = pd.DataFrame(data=adj_matrix,index=samples_name, columns=samples_name)
            lst = adj_df.stack().reset_index()
            lst = lst.rename(columns={lst.columns[0]:"from", lst.columns[1]:"to", lst.columns[2]:"edge_value"})
            lst.to_csv("Salmonella_edges_network.csv")
            print(adj_matrix.shape)
            print(lst.shape)

            T=nx.from_pandas_edgelist(df=lst, source='from', target='to', edge_attr='edge_value', create_using=nx.Graph() )

            # compute the best partition
            partition = community_louvain.best_partition(T)
            partition_df = pd.DataFrame()

            for key in partition.keys():
                partition_df.loc[key,"Community"] = partition[key]

            partition_df.to_csv("Salmonella_communities.csv")

            
            
            carac = metadata_df
            sample_carac = carac.index

            carac = carac.reindex(T.nodes())        
            
            # And I need to transform my categorical column in a numerical value: group1->1, group2->2...
            carac['Sample Host']=pd.Categorical(carac['Sample Host'])
            my_color_node_Type = carac['Sample Host'].cat.codes

            carac['Time']=pd.Categorical(carac['Time'])
            my_color_node_Date = carac['Time'].cat.codes

            carac['Sample Source']=pd.Categorical(carac['Sample Source'])
            my_color_node_Source = carac['Sample Source'].cat.codes

            carac['Phylogroup']=pd.Categorical(carac['Phylogroup'])
            my_color_node_Source = carac['Phylogroup'].cat.codes

            lst['edge_value'] = pd.Categorical(lst['edge_value'])
            my_color_edge = lst['edge_value'].cat.codes
                    
            ColorLegend_Node_Type = {'Chicken': 0,'Environment': 1}
            ColorLegend_Node_Date = {'T1': 0, 'T2': 1, 'T3': 2}
            ColorLegend_Node_Source = {'Chicken faeces': 0,'Chicken carcass': 1,'Chicken cecal': 2, 'Feed': 3,
                'Soil': 4, 'Sewage': 5, 'Chicken feather': 6,'Abattoir environment': 7,'Barn environment': 8,
                'Chicken anal': 9, 'Water': 10}
            ColorLegend_Node_Phylogroup = {'Agona': 0,'Alachua': 1,'Enteritidis': 2, 'Havana': 3,
                'Indiana': 4, 'Kedougou': 5, 'Kentucky': 6,'Mbandaka': 7}
            
            values_edge = []
            for edge in T.edges():
                edge_val = T.get_edge_data(edge[0], edge[1])
                values_edge.append(ColorLegend_Edge[np.ceil(edge_val['edge_value'])])

            values_node_Type = []
            values_node_Date = []
            values_node_Source = []
            values_node_Phylogroup = []
            for node  in T.nodes():
                values_node_Type.append(ColorLegend_Node_Type[carac.loc[node,'Sample Host']])
                values_node_Date.append(ColorLegend_Node_Date[carac.loc[node,'Time']])
                values_node_Source.append(ColorLegend_Node_Source[carac.loc[node,'Sample Source']])
                values_node_Phylogroup.append(ColorLegend_Node_Phylogroup[carac.loc[node,'Phylogroup']])

            values_node_Type = np.array(values_node_Type)
            values_node_Date = np.array(values_node_Date)
            values_node_Source = np.array(values_node_Source)
            values_node_Phylogroup = np.array(values_node_Phylogroup)
            
            print(len(T.nodes()))
            id_node_env = np.where(values_node_Type == 1)[0]
            env_len = len(id_node_env)
            print("env = {}".format(env_len))
            id_node_chicken = np.where(values_node_Type == 0)[0]
            chicken_len = len(id_node_chicken)
            print("chicken = {}".format(chicken_len))
            
            # compute maximum value s.t. all colors can be normalised
            maxval_node_Type = np.max(values_node_Type) 
            maxval_node_Date = np.max(values_node_Date) 
            maxval_node_Source = np.max(values_node_Source) 
            maxval_node_Phylogroup = np.max(values_node_Phylogroup) 
            maxval_edge = np.max(values_edge) 
            
            # get colormap
            cmap_Type=cm.Paired
            cmap_Date=cm.Set1
            cmap_Phylogroup=cm.tab10
            cmap_Source=cm.Paired 

            pos = nx.nx_agraph.graphviz_layout(T)

            nodes = np.array(T.nodes())
 
            ## Nodes Time

            nx.draw_networkx_nodes(T, pos, node_size=15, nodelist= nodes[id_node_chicken], node_color=[cmap_Date(v/maxval_node_Date) for v in values_node_Date[id_node_chicken]], 
                node_shape = 'o', ax=ax[3])
            nx.draw_networkx_nodes(T, pos, node_size=15, nodelist= nodes[id_node_env], node_color=[cmap_Date(v/maxval_node_Date) for v in values_node_Date[id_node_env]], 
                node_shape = '*', ax=ax[3])
            
            nx.draw_networkx_edges(T,pos,alpha = 0.6, edge_color=[cmap_edge(v/maxval_edge) for v in values_edge], edge_cmap=cmap_edge, ax=ax[3], width=0.6)
            legend_elements = []
            for v in set(values_node_Date):
                if v == 0:
                    label = "T1"
                elif v == 1:
                    label = "T2"
                else:
                    label = "T3"
                legend_elements.append(Line2D([], [], marker='o', markeredgecolor=cmap_Date(v/maxval_node_Date), label=label,
                                color = 'w', markerfacecolor = cmap_Date(v/maxval_node_Date), markersize=10))
            
            ax[3].legend(handles = legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.03),
                fancybox=True, shadow=True, ncol=5, fontsize=9)

            ## Nodes Source Type

            nx.draw_networkx_nodes(T, pos, node_size=15, nodelist= nodes[id_node_chicken], node_color=[cmap_Source(v/maxval_node_Source) for v in values_node_Source[id_node_chicken]], 
                node_shape = 'o', ax=ax[4])
            nx.draw_networkx_nodes(T, pos, node_size=15, nodelist= nodes[id_node_env], node_color=[cmap_Source(v/maxval_node_Source) for v in values_node_Source[id_node_env]], 
                node_shape = '*', ax=ax[4])
            
            nx.draw_networkx_edges(T,pos,alpha = 0.6, edge_color=[cmap_edge(v/maxval_edge) for v in values_edge], edge_cmap=cmap_edge, ax=ax[4], width=0.6)
            legend_elements = []
            for v in set(values_node_Source):
                if v == 0:
                    label = "Chicken faeces"
                elif v == 1:
                    label = "Chicken carcass"
                elif v == 2:
                    label = "Chicken cecal"
                elif v == 3:
                    label = "Feed"
                elif v == 4:
                    label = "Soil"
                elif v == 5:
                    label = "Sewage"
                elif v == 6:
                    label = "Chicken feather"
                elif v == 7:
                    label = "Abattoir environment"
                elif v == 8:
                    label = "Barn environment"
                elif v == 9:
                    label = "Chicken anal"
                else:
                    label = "Water"
                legend_elements.append(Line2D([], [], marker='o', markeredgecolor=cmap_Source(v/maxval_node_Source), label=label,
                                color = 'w', markerfacecolor = cmap_Source(v/maxval_node_Source), markersize=10))
            
            ax[4].legend(handles = legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.03),
                fancybox=True, shadow=True, ncol=3, fontsize=9)
            ax[4].set_title("Salmonella enterica", style='italic')

            ## Nodes Phylogroup

            nx.draw_networkx_nodes(T, pos, node_size=15, nodelist= nodes[id_node_chicken], node_color=[cmap_Phylogroup(v/maxval_node_Phylogroup) for v in values_node_Phylogroup[id_node_chicken]], 
                node_shape = 'o', ax=ax[5])
            nx.draw_networkx_nodes(T, pos, node_size=15, nodelist= nodes[id_node_env], node_color=[cmap_Phylogroup(v/maxval_node_Phylogroup) for v in values_node_Phylogroup[id_node_env]], 
                node_shape = '*', ax=ax[5])
            
            nx.draw_networkx_edges(T,pos,alpha = 0.6, edge_color=[cmap_edge(v/maxval_edge) for v in values_edge], edge_cmap=cmap_edge, ax=ax[5], width=0.6)
            legend_elements = []
            for v in set(values_node_Phylogroup):
                if v == 0:
                    label = "Agona"
                elif v == 1:
                    label = "Alachua"
                elif v == 2:
                    label = "Enteritidis"
                elif v == 3:
                    label = "Havana"
                elif v == 4:
                    label = "Indiana"
                elif v == 5:
                    label = "Kedougou"
                elif v == 6:
                    label = "Kentucky"
                elif v == 7:
                    label = "Mbandaka"

                legend_elements.append(Line2D([], [], marker='o', markeredgecolor=cmap_Phylogroup(v/maxval_node_Phylogroup), label=label,
                                color = 'w', markerfacecolor = cmap_Phylogroup(v/maxval_node_Phylogroup), markersize=10))
            
            ax[5].legend(handles = legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.03),
                fancybox=True, shadow=True, ncol=3, fontsize=9)

    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    plt.savefig('SNP_network_WGS_commercial.svg', bbox_inches='tight')
        
        
        
    