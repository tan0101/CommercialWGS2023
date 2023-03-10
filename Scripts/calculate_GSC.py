# -*- coding: utf-8 -*-

from platform import dist
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, _DistanceMatrix
from collections import Counter, OrderedDict
from ete3 import Tree
from scipy import stats
from Bio import Phylo
import Bio
import numpy as np
import pandas as pd
import math
import os
import sys
import matplotlib.pyplot as plt

folder = "EcoliSNPs"
name_dataset = "EC518"
results_folder = "Population Correction"

def get_weights():
    distance_matrix, samples = get_mash_distances()
    dist_mat = distance_matrix_modifier(distance_matrix)
    distance_matrix_to_phyloxml(samples, dist_mat)   
    phyloxml_to_newick(folder+"/"+results_folder+"/tree_xml.txt")
    #tree = Phylo.read(folder+"/"+results_folder+"/tree_xml.txt", "phyloxml")
    #Phylo.draw(tree)
    #input("continue_draw")
    weights = GSC_weights_from_newick(folder+"/"+results_folder+"/tree_newick.txt", normalize="mean1")
    df = pd.DataFrame.from_dict(weights,orient='index', columns=['weights'])
    df = df.reindex(samples)
    
    return df

def get_mash_distances():
    distance_matrix = pd.read_excel(folder+"/"+name_dataset+'_distancematrix.xlsx', header=[0], index_col=[0])
    return distance_matrix.values.tolist(), list(distance_matrix.columns)

def distance_matrix_modifier(distancematrix):
    # Modifies distance matrix to be suitable argument 
    # for Bio.Phylo.TreeConstruction._DistanceMatrix function
    for i in range(len(distancematrix)):
        for j in range(len(distancematrix[i])):
            distancematrix[i][j] = float(distancematrix[i][j])
    distance_matrix = []
    counter = 1
    for i in range(len(distancematrix)):
        data = distancematrix[i]
        distance_matrix.append(data[0:counter])
        counter += 1

    return(distance_matrix)

def distance_matrix_to_phyloxml(samples_order, distance_matrix):
    #Converting distance matrix to phyloxml
    dm = _DistanceMatrix(samples_order, distance_matrix)
    tree_xml = DistanceTreeConstructor().nj(dm)
    with open(folder+"/"+results_folder+"/tree_xml.txt", "w+") as f1:
        Bio.Phylo.write(tree_xml, f1, "phyloxml")

def phyloxml_to_newick(phyloxml):
    #Converting phyloxml to newick
    with open(folder+"/"+results_folder+"/tree_newick.txt", "w+") as f1:
        Bio.Phylo.convert(phyloxml, "phyloxml", f1, "newick")

def GSC_weights_from_newick(newick_tree, normalize=None):
    # Calculating Gerstein Sonnhammer Coathia weights from Newick 
    # string. Returns dictionary where sample names are keys and GSC 
    # weights are values.
    tree = Tree(newick_tree, format=1)
    tree = clip_branch_lengths(tree)
    set_branch_sum(tree)
    set_node_weight(tree)

    tree.show()
    
    weights = {}
    for leaf in tree.iter_leaves():
        weights[leaf.name] = leaf.NodeWeight
    if normalize == "mean1":
        weights = {k: v*len(weights) for k, v in weights.items()}
    return(weights)

def clip_branch_lengths(tree, min_val=1e-9, max_val=1e9): 
    for branch in tree.traverse("levelorder"):
        if branch.dist > max_val:
            branch.dist = max_val
        elif branch.dist < min_val:
            branch.dist = min_val
    
    return tree

def set_branch_sum(tree):
    total = 0
    for child in tree.get_children():
        tree_child = set_branch_sum(child)
        total += tree_child.BranchSum
        total += tree_child.dist
        
    tree.BranchSum = total

    return tree

def set_node_weight(tree):
    parent = tree.up
    if parent is None:
        tree.NodeWeight = 1.0
    else:
        tree.NodeWeight = parent.NodeWeight * (tree.dist + tree.BranchSum)/parent.BranchSum

    for child in tree.get_children():
        tree = set_node_weight(child)

    return tree

if __name__ == "__main__":
    if not os.path.exists(folder+'/'+results_folder):
        os.makedirs(folder+'/'+results_folder)

    weights = get_weights()
    
    weights.to_csv(folder+'/'+results_folder+'/GSC_weights_'+name_dataset+'.csv', index_label=['ID'])
        
