
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
from lingpy.compare.strings import bisim1


def similarity_function_tup(tup):
    a,b = tup
    return similarity_function(a,b)

def normalize_word(w):
    return w.lower()


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

    pair_sim = np.array([similarity_function(word,t) for t in tuples])
    if kernel == "poly":        
        k = (np.ones(len(pair_sim)) - pair_sim)**hyperparam
    else:              
        k = np.exp(-hyperparam * (pair_sim**2))             

    return k.dot(alphas_lambdas_div)


'''
Parsing user arguments
'''

argParser = argparse.ArgumentParser(description="KPCA embeddings training script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argParser.add_argument('--repr', type=str,help="Representative words (use a subset of your vocabulary if it is too large for your memory restrictions)", action='store',required=False)
argParser.add_argument('--vocab', type=str,help="Vocabulary path", action='store',required=True)
argParser.add_argument('--sim', type=str,help="Similarity function: 'bisim1' (bigram similarity),'sorensen_plus' (Sørensen–Dice index for n-grams )(default: 'sorensen_plus' )", action='store',required=False, default = "sorensen_plus")
argParser.add_argument('--kernel', type=str,help="Kernel: 'poly','rbf' (default: 'poly' )", action='store',required=False, default = "poly")
argParser.add_argument('--hyperparam', type=float,help="Hyperparameter for the selected kernel: sigma for RBF kernel and the degree for the polynomial kernel (default 2)", action='store',required=False, default = 2)
argParser.add_argument('--max_ngram', type=int,help="Maximum length of the n-grams considered by the bigram similarity function (default 2)", action='store',required=False, default = 2)
argParser.add_argument('--size', type=int,help="Number of principal components of the embeddings (default 1500)", action='store',required=False, default= 1500)
argParser.add_argument('--cores', type=int,help="Number of processes to be started for computation (default: number of available cores)", action='store',required=False, default= multiprocessing.cpu_count())
argParser.add_argument('--output', type=str,help="Output folder for the KPCA embeddings (default: current folder)", action='store',required=False, default= ".")
args = argParser.parse_args()


MAX_NGRAM = args.max_ngram
n_components = args.size
reprPath = args.repr
vocabPath = args.vocab
kernel = args.kernel
hyperparam = args.hyperparam
cores = args.cores
outputPath = args.output

#Similarity function to be used as dot product for KPCA
similarity_function = eval(args.sim)

if reprPath == None:
    reprPath = vocabPath

'''
Preprocessing

'''
with codecs.open(reprPath, "r") as fIn:
        reprVocab = [  normalize_word(w[:-1]) for w in fIn if len(w[:-1].split()) ==1]

termcolor.cprint("Generating word pairs\n", "blue")
reprVocabLen = len(reprVocab)

pairsArray = np.array([ (t1,t2) for t1 in reprVocab for t2 in reprVocab])

pool = multiprocessing.Pool(processes=cores)



'''
Similarity matrix computation: the similarity of all word pairs from the representative words is computed
'''
termcolor.cprint("Computing similarity matrix\n", "blue")

simMatrix = np.array(pool.map(similarity_function_tup, pairsArray )).reshape(reprVocabLen, reprVocabLen)

pairsArray = None


'''
Kernel Principal Component Analysis
'''
termcolor.cprint("Solving eigevector/eigenvalues problem\n", "blue")

if kernel == "rbf":    
    K = np.exp(-hyperparam * (simMatrix**2))
else: #poly
    distMatrix = np.ones(len(simMatrix))- simMatrix
    K = distMatrix**hyperparam


# Centering the symmetric NxN kernel matrix.
N = K.shape[0]
one_n = np.ones((N,N)) / N
K_norm = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
# Obtaining eigenvalues in descending order with corresponding eigenvectors from the symmetric matrix.    
eigvals, eigvecs = eigh(K_norm)

 
alphas = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
lambdas = [eigvals[-i] for i in range(1,n_components+1)]


pickle.dump( alphas, open( outputPath+"/alphas_{}_{}_{}_{}_{}.p".format(similarity_function.__name__, len(reprVocab),kernel, hyperparam, n_components), "wb" ) )
pickle.dump( lambdas, open( outputPath+"/lambdas_{}_{}_{}_{}_{}.p".format(similarity_function.__name__, len(reprVocab),kernel, hyperparam, n_components), "wb" ) )     


'''
Projection to KPCA embeddings of the vocabulary
'''
with codecs.open(vocabPath, "r") as fIn:
    vocab = [ normalize_word(w[:-1]) for w in fIn if len(w[:-1].split()) ==1]

termcolor.cprint("Projecting known vocabulary to KPCA embeddings\n", "blue")

#X_train = pool.map(projectWordTup, [(word,reprVocab, hyperparam,alphas, lambdas, kernel) for word in vocab] )  
alphas_lambdas_div = alphas / lambdas
X_train = pool.map(projectWordTup, [(word,reprVocab, hyperparam, alphas_lambdas_div, kernel) for word in vocab] )  

pickle.dump( X_train, open(outputPath+"/KPCA_{}_{}_{}_{}_{}.p".format(similarity_function.__name__, len(reprVocab),kernel,hyperparam, n_components), "wb" ) )

