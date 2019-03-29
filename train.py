from hyper import *
import argparse
import random
import pickle
import math
import numpy as np
import re

def sanitizeWord(w):
    if w.startswith("$T"):
        return w
    if w == ("xTHEMEx"):
        return "xTHEMEx"
    if w == ("xTRIGGERx"):
        return "xTRIGGERx"
    w = w.lower()
    if is_number(w):
        return "xnumx"
    w = re.sub("[^a-z_]+","",w)
    if w:
        return w

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def load_embeddings(file):
    emb_matrix = None
    emb_dict = dict()
    for line in open(file):
        if not len(line.split()) == 2:
            if "\t" in line:
                delimiter = "\t"
            else:
                delimiter = " "
            line_split = line.rstrip().split(delimiter)
            # extract word and vector
            word = line_split[0]
            vector = np.array([float(i) for i in line_split[1:]])
            embedding_size = vector.shape[0]
            emb_dict[word] = vector
    return emb_dict

if __name__ == '__main__':
    random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('pos')
    parser.add_argument('neg')
    args = parser.parse_args()
    
    words = ["UNKNOWN"]
    w2i = {"UNKNOWN":0}
    candidates = list()
    for line in open("SemEval2018-Task9/vocabulary/1A.english.vocabulary.txt"):
        temp = line.strip().lower().split(" ")
        candidates.append(temp)
        for t in temp:
            t = sanitizeWord(t)
            if t and t not in w2i:
                w2i[t] = len(words)
                words.append(t)
    embeds = load_embeddings("glove.6B.200d.txt")
    trainning_set = list()
    for line in open(args.pos):
        pairs = line.strip().split("\t")
        for p in pairs:
            p = p.lower().strip().replace("(","").replace(")","")
            q = p.split(",")[0]
            h = p.split(",")[1]
            trainning_set.append((q, h, 1))
    for line in open(args.neg):
        pairs = line.strip().split("\t")
        for p in pairs:
            p = p.lower().strip().replace("(","").replace(")","")
            q = p.split(",")[0]
            h = p.split(",")[1]
            trainning_set.append((q, h, 0))
    print ("Trainning")
    
    model = Hyper(words, w2i, embeds, 200, 200, 24)
    for i in range(200):
        random.shuffle(trainning_set)
        model.train(trainning_set)
        if ((i) % 10) == 0:
            ehs = list()
            for ws in candidates:
                eh = dy.average([model.word_embeddings[model.w2i[w]] if w in model.w2i else model.word_embeddings[0] for w in ws])
                ehs.append(eh)
            ehs = dy.concatenate_cols(ehs)
            with open("result"+str(i)+".txt", "w") as f:
                for line in open("SemEval2018-Task9/test/data/1A.english.test.data.txt"):
                    query = line.strip().lower().split("\t")[0]
                    f.write(query)
                    query = query.split(" ")
                    for c in model.get_hypers(query, candidates, ehs, w2i):
                        c = ' '.join(c)
                        f.write("\t"+c)
                    f.write("\n")
            model.save("model"+str(i))


