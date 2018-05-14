#   Copyright 2018 Fraunhofer IAIS
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import multiprocessing
from scipy.linalg import eigh
import codecs
import pickle
import distance
from os.path import expanduser
import termcolor
import argparse
import pyximport; pyximport.install()
from similarity import ngram_sim



def similarity_function_tup(tup):
    a,b = tup
    return similarity_function(a,b)

def distance_function_tup(tup):
    a,b = tup
    return 1-similarity_function(a,b)

def normalize_word(w, lower = True):
    if lower:
        return w.lower()
    else:
        return w


def ngrams(s, n):
    string = " "+s+" "
    return list(set([string[i:i+n] for i in range(len(string)-n+1)]))

#This function takes two words
def sorensen_plus(a,b):     
    ng1 = [ngrams(a, i) for i in range(1,min(len(a),len(b))+1)]
    ng2 = [ngrams(b, i) for i in range(1,min(len(a),len(b))+1)]
    N = min(len(ng1),len(ng2))            
    return 1 - np.sum(distance.sorensen(ng1[i], ng2[i]) for i in range(N))/ N

def projectWordTup(tup):
    word    = tup[0]
    tuples  = tup[1]
    hyperparam   = tup[2]
    alphas_lambdas_div  = tup[3]
    kernel = tup[4]

    if kernel == "poly": #Polynomial kernel       
        pair_sim = np.array([similarity_function(word,t) for t in tuples])
        
        k = pair_sim**hyperparam

    else: #RBF kernel
        pair_dist = np.array([1-similarity_function(word,t) for t in tuples])
        k = np.exp(-hyperparam * (pair_dist**2))             

    return k.dot(alphas_lambdas_div)    


def projectWordsGPU(vocab,reprVocab, hyperparam, alphas_lambdas_div, kernel, simMatrix= None):
    pairsArray = [ (t1,t2) for t1 in vocab for t2 in reprVocab]
    
    if kernel == "poly":        
        if simMatrix == None:
            simMatrix = np.array(pool.map(similarity_function_tup, pairsArray )).reshape(len(vocab), len(reprVocab))
        termcolor.cprint("Similarity matrix computed\n", "blue")
        k =  cm.pow(cm.CUDAMatrix(simMatrix),hyperparam)
    else:              
        if simMatrix== None:
            distMatrix = np.array(pool.map(distance_function_tup, pairsArray )).reshape(len(vocab), len(reprVocab))
        else:
            distMatrix = 1 - simMatrix
        termcolor.cprint("Distance matrix computed\n", "blue")
        k = cm.exp(-hyperparam * (cm.pow(cm.CUDAMatrix(distMatrix),2)))                
    return cm.dot(k,cm.CUDAMatrix(alphas_lambdas_div)).asarray()

def computeSimMatrix(reprVocab):    
    return np.array(pool.map(similarity_function_tup, [ (t1,t2) for t1 in reprVocab for t2 in reprVocab] )).reshape(len(reprVocab), len(reprVocab))

'''
Parsing user arguments
'''

argParser = argparse.ArgumentParser(description="KPCA embeddings training script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argParser.add_argument('--repr', type=str,help="Representative words (use a subset of your vocabulary if it is too large for your memory restrictions)", action='store',required=False)
argParser.add_argument('--vocab', type=str,help="Vocabulary path", action='store',required=True)
argParser.add_argument('--sim', type=str,help="Similarity function: 'ngram_sim' (n-gram similarity),'sorensen_plus' (Sørensen–Dice index for n-grams )(default: 'ngram_sim' )", action='store',required=False, default = "ngram_sim")
argParser.add_argument('--n_ngram', type=str,help="Size of the n-grams when n-gram similarity function is used (default: 2)", action='store',required=False, default = 2)
argParser.add_argument('--kernel', type=str,help="Kernel: 'poly','rbf' (default: 'poly' )", action='store',required=False, default = "poly")
argParser.add_argument('--hyperparam', type=float,help="Hyperparameter for the selected kernel: sigma for RBF kernel and the degree for the polynomial kernel (default 2)", action='store',required=False, default = 2)
argParser.add_argument('--size', type=int,help="Number of principal components of the embeddings (default 1500)", action='store',required=False, default= 1500)
argParser.add_argument('--cores', type=int,help="Number of processes to be started for computation (default: number of available cores)", action='store',required=False, default= multiprocessing.cpu_count())
argParser.add_argument('--output', type=str,help="Output folder for the KPCA embeddings (default: current folder)", action='store',required=False, default= ".")
argParser.add_argument('--use_gpu', type=bool,help="Output folder for the KPCA embeddings (default: current folder)", action='store',required=False, default= False)
argParser.add_argument('--sim_matrix', type=str,help="'compute' for computing the similarity matrix only, 'infer' for loading a precomputed matrix and infer KPCA embeddings, 'full' for computing all (default 'full') ", action='store',required=False, default= 'full')
args = argParser.parse_args()



n_components = args.size
reprPath = args.repr
vocabPath = args.vocab
kernel = args.kernel
hyperparam = args.hyperparam
cores = args.cores
outputPath = args.output
useGPU = args.use_gpu
simMatrixMode = args.sim_matrix

#Similarity function to be used as dot product for KPCA
def similarity_function(a,b):
    if args.sim == "ngram_sim":
        return 1-eval(args.sim)(a,b,args.n_ngram)
    else:
        return 1-eval(args.sim)(a,b)

if useGPU:
    import cudamat as cm
    cm.cublas_init()

if reprPath == None:
    reprPath = vocabPath

pool = multiprocessing.Pool(processes=cores)

'''
Preprocessing

'''
with codecs.open(reprPath, "r") as fIn:
    reprVocab = [  normalize_word(w[:-1]) for w in fIn if len(w[:-1].split()) ==1]

'''
Similarity matrix computation: the similarity of all word pairs from the representative words is computed
'''
if simMatrixMode == "infer":
    simMatrix = pickle.load( open(outputPath+"/simMatrix_{}_{}_{}_{}.p".format(eval(args.sim).__name__, len(reprVocab),kernel,hyperparam), "rb" ) )
else:
    termcolor.cprint("Computing similarity matrix\n", "blue")
    simMatrix = computeSimMatrix(reprVocab)
    if simMatrixMode == "compute":
        pickle.dump( simMatrix, open(outputPath+"/simMatrix_{}_{}_{}_{}.p".format(eval(args.sim).__name__, len(reprVocab),kernel,hyperparam), "wb" ) )
        exit()

termcolor.cprint("Generating word pairs\n", "blue")
reprVocabLen = len(reprVocab)

pairsArray = np.array([ (t1,t2) for t1 in reprVocab for t2 in reprVocab])



'''
Kernel Principal Component Analysis
'''
termcolor.cprint("Solving eigevector/eigenvalues problem\n", "blue")

if kernel == "rbf":    
    distMatrix = np.ones(len(simMatrix))- simMatrix
    K = np.exp(-hyperparam * (distMatrix**2))
else: #poly    
    K = simMatrix**hyperparam


# Centering the symmetric NxN kernel matrix.
N = K.shape[0]
one_n = np.ones((N,N)) / N
K_norm = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
# Obtaining eigenvalues in descending order with corresponding eigenvectors from the symmetric matrix.    

eigvals, eigvecs = eigh(K_norm)

 
alphas = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
lambdas = [eigvals[-i] for i in range(1,n_components+1)]


pickle.dump( alphas, open( outputPath+"/alphas_{}_{}_{}_{}_{}.p".format(eval(args.sim).__name__, len(reprVocab),kernel, hyperparam, n_components), "wb" ) )
pickle.dump( lambdas, open( outputPath+"/lambdas_{}_{}_{}_{}_{}.p".format(eval(args.sim).__name__, len(reprVocab),kernel, hyperparam, n_components), "wb" ) )     


'''
Projection to KPCA embeddings of the vocabulary
'''
with codecs.open(vocabPath, "r") as fIn:
    vocab = [ normalize_word(w[:-1]) for w in fIn if len(w[:-1].split()) ==1]

termcolor.cprint("Projecting known vocabulary to KPCA embeddings\n", "blue")

alphas_lambdas_div = alphas / lambdas

if useGPU:
    X_train = projectWordsGPU( vocab,reprVocab, hyperparam, alphas_lambdas_div, kernel)
    cm.shutdown ()
else:
    X_train = pool.map(projectWordTup, [(word,reprVocab, hyperparam, alphas_lambdas_div, kernel) for word in vocab] )  

pickle.dump( X_train, open(outputPath+"/KPCA_{}_{}_{}_{}_{}.p".format(eval(args.sim).__name__, len(reprVocab),kernel,hyperparam, n_components), "wb" ) )

