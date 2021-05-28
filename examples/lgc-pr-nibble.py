#!/usr/bin/env python

"""
    lgc/main.py
    
    Note to program performers:
        - parallel_pr_nibble produces the same results as ligra's `apps/localAlg/ACL-Sync-Local-Opt.C`
        - ista produces the same results as LocalGraphClustering's `ista_dinput_dense` method
"""

import os
import sys
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from scipy import sparse
from scipy.io import mmread
from scipy.stats import spearmanr

# --
# Parallel PR-Nibble

def parallel_pr_nibble(seeds, degrees, num_nodes, adj_indices, adj_indptr, alpha, epsilon):
    out = []
    for seed in tqdm(seeds):
        p = np.zeros(num_nodes)
        r = np.zeros(num_nodes)
        r[seed] = 1
        
        frontier = np.array([seed])
        while True:
            if len(frontier) == 0:
                break
            
            r_prime = r.copy()
            for node_idx in frontier:
                p[node_idx] += (2 * alpha) / (1 + alpha) * r[node_idx]
                r_prime[node_idx] = 0
           
            for src_idx in frontier:
                neighbors = adj_indices[adj_indptr[src_idx]:adj_indptr[src_idx + 1]]
                for dst_idx in neighbors:
                    update = ((1 - alpha) / (1 + alpha)) * r[src_idx] / degrees[src_idx]
                    r_prime[dst_idx] += update
                    
            r = r_prime
            
            frontier = np.where((r >= degrees * epsilon) & (degrees > 0))[0]
        
        out.append(p)
    
    return np.column_stack(out)

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-seeds', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--pnib-epsilon', type=float, default=1e-6)
    args = parser.parse_args()
    
    # !! In order to check accuracy, you _must_ use these parameters !!
    assert args.num_seeds == 50
    assert args.alpha == 0.15
    assert args.pnib_epsilon == 1e-6
    
    return args



args = parse_args()

adj = mmread('data/jhu.mtx').tocsr()
degrees = np.asarray(adj.sum(axis=-1)).squeeze().astype(int)
num_nodes = adj.shape[0]
adj_indices = adj.indices
adj_indptr = adj.indptr

pnib_seeds = np.array(range(args.num_seeds))

alpha = args.alpha
pnib_epsilon = args.pnib_epsilon

t = time()
pnib_scores = parallel_pr_nibble(pnib_seeds, degrees, num_nodes, adj_indices, adj_indptr, alpha=alpha, epsilon=pnib_epsilon)
t2 = time()
assert pnib_scores.shape[0] == adj.shape[0]
assert pnib_scores.shape[1] == len(pnib_seeds)
pnib_elapsed = time() - t

print("[Nibble Elapsed Time]: ", (t2 - t))

os.makedirs('results', exist_ok=True)

np.savetxt('results/pnib_score.txt', pnib_scores)

open('results/pnib_elapsed', 'w').write(str(pnib_elapsed))

