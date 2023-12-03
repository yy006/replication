#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:10:42 2016
"""

# main script for time CD 
import numpy as np
import util_timeCD as util
import pickle

# PARAMETERS

nw = 20936 # number of words in vocab (11068100/20936 for ngram/nyt)
T = range(1990, 2016) # total number of time points (20/range(27) for ngram/nyt)
cuda = True

trainhead = 'data/wordPairPMI_' # location of training data
savehead = 'results/'
    
def print_params(r, lam, tau, gam, emph, ITERS):
    print('rank = {}'.format(r))
    print('frob regularizer = {}'.format(lam))
    print('time regularizer = {}'.format(tau))
    print('symmetry regularizer = {}'.format(gam))
    print('emphasize param = {}'.format(emph))
    print('total iterations = {}'.format(ITERS))
    
if __name__ == '__main__':
    import sys
    ITERS = 5 # total passes over the data
    lam = 10 # frob regularizer
    gam = 100 # forcing regularizer
    tau = 50 # smoothing regularizer
    r = 50 # rank
    b = nw # batch size
    emph = 1 # emphasize the nonzero

    foo = sys.argv
    for i in range(1, len(foo)):
        if foo[i] == '-r': r = int(float(foo[i+1]))
        if foo[i] == '-iters': ITERS = int(float(foo[i+1]))            
        if foo[i] == '-lam': lam = float(foo[i+1])
        if foo[i] == '-tau': tau = float(foo[i+1])
        if foo[i] == '-gam': gam = float(foo[i+1])
        if foo[i] == '-b': b = int(float(foo[i+1]))
        if foo[i] == '-emph': emph = float(foo[i+1])
        if foo[i] == '-check': erchk = foo[i+1]
    
    savefile = f'{savehead}L{lam}T{tau}G{gam}A{emph}'
    
    print('starting training with the following parameters')
    print_params(r, lam, tau, gam, emph, ITERS)
    print('there are a total of {} words, and {} time points'.format(nw, len(T)))
    
    print('X*X*X*X*X*X*X*X*X')
    print('initializing')
    
    Ulist, Vlist = util.import_static_init(T)
    print(Ulist)
    print(Vlist)
    print('getting batch indices')
    if b < nw:
        b_ind = util.getbatches(nw, b)
    else:
        b_ind = [range(nw)]
    
    import time
    start_time = time.time()
    # sequential updates
    for iteration in range(ITERS):
        print_params(r, lam, tau, gam, emph, ITERS)
        try:
            Ulist = pickle.load(open(f"{savefile}ngU_iter{iteration}.p", "rb"))
            Vlist = pickle.load(open(f"{savefile}ngV_iter{iteration}.p", "rb"))
            print(f'iteration {iteration} loaded successfully')
            continue
        except IOError:
            pass

        # shuffle times
        if iteration == 0:
            times = T
        else:
            times = np.random.permutation(T)
        
        for t in range(len(times)): # select a time
            print(f'iteration {iteration}, time {t}')
            f = trainhead + str(t) + '.csv'
            print(f)
            
            pmi = util.getmat(f, nw, False)
            for j in range(len(b_ind)): # select a mini batch
                print(f'{j} out of {len(b_ind)}')
                ind = b_ind[j]

                # Update logic...

            print('time elapsed = ', time.time() - start_time)

            pickle.dump(Ulist, open(f"{savefile}ngU_iter{iteration}.p", "wb"), pickle.HIGHEST_PROTOCOL)
            pickle.dump(Vlist, open(f"{savefile}ngV_iter{iteration}.p", "wb"), pickle.HIGHEST_PROTOCOL)
