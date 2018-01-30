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

from sklearn import linear_model, decomposition, datasets, neighbors
from scipy.spatial.distance import pdist, squareform
import logging
import os,sys
import codecs
import pickle
import distance
from unidecode import unidecode
import scipy
import json
from os.path import expanduser
import lingpy.compare.strings
import re
from num2words import num2words


#TODO: optimize this parameter

DISTANCE_THRESHOLD = 0.2

def regexMatch(pattern,text):
    '''
        A function that tests if a regular expression is met in a given string
        Example:
        >>> pattern=".*20[0-9]{2}"
        >>> regexMatch(pattern,"This year is 1999")
            False

        >>> regexMatch(pattern,"last year was the famous year 2000")
            True
        >>> regexMatch(pattern,"the year 2000 was the milenium")
            True
    '''
    regexResult=re.match(pattern,text)
    if regexResult==None:
        return False
    else:
        return True


def ngrams(s, n):
    string = " "+s+" "
    return list(set([string[i:i+n] for i in range(len(string)-n+1)]))


# In[3]:

def similarity(s1,s2):
    ngrams1 = [ngrams(s1,j) for j in range(2,  min( len(s1)+1 , MAX_NGRAM))  ]  
    ngrams2 = [ngrams(s2,j) for j in range(2,  min( len(s2)+1 , MAX_NGRAM))  ]  
    N = min(len(ngrams1),len(ngrams2))+1        
    ngramMatches = np.array([len(np.intersect1d(ngrams1[i], ngrams2[i])) for i in range(N-1)])
    ngramLens1 = np.array([len(ngrams1[i] ) for i in range(N-1)] )
    ngramLens2 = np.array([len(ngrams2[i] ) for i in range(N-1)] )

    ngramMatchesLen = len(ngramMatches)
    if ngramMatchesLen > 0:
        return sum(2* ngramMatches / (ngramLens1 +ngramLens2) )/ ngramMatchesLen
    else:
        return 0


def similarity_ngram(tup):
    ngrams1, ngrams2 = tup
    N = min(len(ngrams1),len(ngrams2))+1        
    ngramMatches = np.array([len(np.intersect1d(ngrams1[i], ngrams2[i])) for i in range(N-1)])
    ngramLens1 = np.array([len(ngrams1[i] ) for i in range(N-1)] )
    ngramLens2 = np.array([len(ngrams2[i] ) for i in range(N-1)] )

    ngramMatchesLen = len(ngramMatches)
    if ngramMatchesLen > 0:
        return sum(2* ngramMatches / (ngramLens1 +ngramLens2) )/ ngramMatchesLen
    else:
        return 0

def projectWordTup(tup):
    word    = tup[0]
    tokens  = tup[1]
    gamma   = tup[2]
    alphas  = tup[3]
    lambdas = tup[4]
    kernel = tup[5]
    
    
    pair_dist = np.array([similarity(word,t) for t in tokens])
    if kernel == "poly":        
        k = pair_dist**gamma
    else:        
        k = np.exp(-gamma * ((np.ones(len(pair_dist)) - pair_dist)**2))             
    
    return k.dot(alphas / lambdas)

def similarity_function_tup(tup):
    a,b = tup
    return similarity_function(a,b)

def normalize_word(w):
    return unidecode(w.lower())

def projectWord(word , degree=2):
    pair_dist = np.array([ 1- similarity_function(word,t) for t in maxoids])           
    k = pair_dist**degree
    return k.dot(alphas / lambdas)

def most_similar(w, topn=10):
    order= 1
    wordTuple = [ ngrams(w,j) for j in range(2,  min( len(w)+1 , MAX_NGRAM) + 1)    ]
    v1 = projectWord(w)    
    D = np.array([(t, scipy.spatial.distance.cosine(v1, kpcaModel[t])  ) for t in kpcaModel ])    

    D_sorted= sorted(D, key=lambda tup: float(tup[1]))

    
    return D_sorted[:topn]

def detectRunOnError(word, vocab):
    for i in range(len(word)-1):
        w1 = word[:i+1]
        w2 = word[i+1:]
        
        if  w1 in vocab  and  w2 in vocab:
            return w1,w2
    return w1, ""


#HOME = expanduser("~")
HOME = "../"



MAX_NGRAM = 2

#sys.path.append("../lingpy") 

similarity_function = lingpy.compare.strings.bisim1
sim_func = "bisim1"
vocabSize = 3000
n_components = 2000

with codecs.open(HOME+"/data/freq_vocab_nl_{}.txt".format(vocabSize), "r") as fIn:
        maxoids = [ normalize_word(w[:-1]) for w in fIn if len(w[:-1].split()) ==1]       

vocabPath = HOME+"/data/nl_wordlist.txt"
with codecs.open(vocabPath, "r") as fIn:
    #vocab = [normalize_word(w[:-1]) for w in fIn if len(w[:-1].split()) ==1]
    vocab = [w[:-1] for w in fIn if len(w[:-1].split()) ==1]

with open(HOME+'/data/KPCA_freq_vocab_nl_{}_{}_poly_2_{}.p'.format(sim_func,vocabSize,n_components), 'rb') as f:
	X_train = pickle.load(f)

kpcaModel = {}
for i in range(len(vocab)):
    kpcaModel[vocab[i]] = X_train[i]

maxoidTuples =  [ [   ngrams(maxoids[i],j) for j in range(2,  min( len(maxoids[i])+1 , MAX_NGRAM) + 1)    ] for i in range(len(maxoids)) ]      

with open(HOME+'/data/alphas_freq_vocab_nl_{}_{}_poly_2_{}.p'.format(sim_func,vocabSize, n_components), 'rb') as f:
    alphas = pickle.load(f)
with open(HOME+'/data/lambdas_freq_vocab_nl_{}_{}_poly_2_{}.p'.format(sim_func,vocabSize, n_components), 'rb') as f:
    lambdas = pickle.load(f)         


jsonPath = HOME+ "/data/test/"
resultPath = HOME+ "/results/"
correctionDict = {}
wordsDict = {}
previousWord = None
for filename in os.listdir(jsonPath):
#for filename in ["page1062.json"]:
    if filename[-5:] == ".json":        
        with codecs.open(jsonPath+filename, "r", encoding="utf-8") as inFile:
            with codecs.open(resultPath+filename, "w", encoding="utf-8") as outFile:
                try:
                    jsonObj = json.loads("\n".join(inFile.readlines()))            
                except:
                    continue
                corrections = []
                isFirstWord = True #Flag to mark start of sentence
                for w in jsonObj["words"]:
                    word = w["text"]                    
                    correction = {}
                    correction["span"] = [w["id"]]
                    distance = 0
                    #Punctuation
                    if regexMatch("^[\.,\!\?;:%\"&\(\\)\[\]\+\-\*\/\#\<\>\-\§\=\']+$", word):
                        if word in [".","!", "?", "..."]:
                            isFirstWord = True
                        continue
                    #Number with dots and commas in between   
                    elif regexMatch("-{0,1}(([0-9]+|([0-9]){1,3}(\.[0-9]{3})+)(,[0-9]+){0,1})$", word):
                        isFirstWord = False
                        continue
                    #Cardinal numbers 
                    elif regexMatch("^([0-9]{1,2}|[1-9]0+) $", word): 
                        closestCorrection = num2words(int(word), lang="nl")
                    #Ordinal numbers
                    elif regexMatch("^[0-9]+(e|de|te|ste)$", word):
                        closestCorrection = num2words(int(re.sub("[a-z]+$", "", word)), lang="nl", to="ordinal")
                    #Alphanumeric words with dashes in between
                    elif word.replace("-","").isalnum() and not word.replace("-","").isalpha():
                        isFirstWord = False
                        continue
                    #We assume we "just" ignore accents, umlauts and stuff like that
                    elif unidecode(word) in vocab:
                        isFirstWord = False
                        continue
                    #We only correct words without weird punctuation
                    elif not word.replace("-","").replace("'","").isalpha():
                        isFirstWord = False
                        continue
                    #Abbreviations
                    elif regexMatch("^([a-zA-Z]\.)+$", word): 
                        isFirstWord = False
                        continue                    
                    
                    
                    
                    
                    #Word not in vocabulary
                    elif word not in vocab and word.lower() not in vocab:       
                            result = most_similar(normalize_word(word), topn=1)[0]
                            closestCorrection = result[0]                            
                            distance = float(result[1])
                            if closestCorrection not in vocab: # Workaround for uppercasing
                                closestCorrection[0] = closestCorrection[0].upper()
                            if distance > DISTANCE_THRESHOLD: #Discard correction, potential proper noun or runon
                                distance = 0
                                #Try runon error
                                w1,w2 = detectRunOnError(word, vocab)
                                if w2 != "": #Runon error detected
                                    closestCorrection = w1 + " "+ w2                                  
                                else:
                                    isFirstWord = False
                                    previousWord = word
                                    continue

                    #Uppercasing required
                    elif word.lower() in vocab and word not in vocab:
                        if isFirstWord:
                            isFirstWord = False
                            continue # Is a valid word because of starting a sentence
                        else:
                            closestCorrection = word.lower()                            
                    #TODO: handling of non-ASCII like äüö                    
                    #No correction required
                    else:
                        isFirstWord = False
                        continue
                    correction["text"] = closestCorrection
                    corrections.append(correction)
                    isFirstWord = False
                    previousWord = None
                    print("Original: {}".format(word))
                    print("Correction: {} ({})".format(closestCorrection, distance))

                json.dump({"corrections": corrections}, outFile)

                        
