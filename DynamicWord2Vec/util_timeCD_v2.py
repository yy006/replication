#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:11:23 2016

@author: raon
"""

import numpy as np
import scipy.io as sio
import copy
import pandas as pd
import scipy.sparse as ss
from sklearn.metrics.pairwise import cosine_similarity

def update(U, Y, Vm1, Vp1, lam, tau, gam, ind, iflag):
    UtU = np.dot(U.T, U)  # rxr
    r = UtU.shape[0]
    if iflag:
        M = UtU + (lam + 2 * tau + gam) * np.eye(r)
    else:
        M = UtU + (lam + tau + gam) * np.eye(r)

    Uty = np.dot(U.T, Y)  # rxb
    Ub = U[ind, :].T  # rxb
    A = Uty + gam * Ub + tau * (Vm1.T + Vp1.T)  # rxb
    Vhat = np.linalg.lstsq(M, A, rcond=None)  # rxb
    return Vhat[0].T  # bxr

def import_static_init(T):
    emb = sio.loadmat('data/emb_static.mat')['emb']
    U = [copy.deepcopy(emb) for _ in T]
    V = [copy.deepcopy(emb) for _ in T]
    return U, V

def initvars(vocab_size, T, rank):
    U, V = [], []
    U.append(np.random.randn(vocab_size, rank) / np.sqrt(rank))
    V.append(np.random.randn(vocab_size, rank) / np.sqrt(rank))
    for t in range(1, T):
        U.append(U[0].copy())
        V.append(V[0].copy())
        print(t)
    return U, V

def getmat(f, v, rowflag):
    data = pd.read_csv(f)
    data = data.to_numpy()

    X = ss.coo_matrix((data[:, 2], (data[:, 0], data[:, 1])), shape=(v, v))

    if rowflag:
        X = ss.csr_matrix(X)
    else:
        X = ss.csc_matrix(X)

    return X

def getbatches(vocab, b):
    batchinds = []
    current = 0
    while current < vocab:
        inds = range(current, min(current + b, vocab))
        current = min(current + b, vocab)
        batchinds.append(inds)
    return batchinds

def getclosest(wid, U):
    C = []
    for t in range(len(U)):
        temp = U[t]
        K = cosine_similarity(temp[wid, :].reshape(1, -1), temp)
        mxinds = np.argsort(-K)
        mxinds = mxinds[0, :10]
        C.append(mxinds)
    return C

def compute_symscore(U, V):
    return np.linalg.norm(U - V)**2

def compute_smoothscore(U, Um1, Up1):
    X = np.linalg.norm(U - Up1)**2 + np.linalg.norm(U - Um1)**2
    return X
