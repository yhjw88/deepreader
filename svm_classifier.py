import json
import unicodedata
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import time

start_time=time.time()

tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
					  if unicodedata.category(unichr(i)).startswith('P'))
def remove_punctuation(text):
	return text.translate(tbl)

with open('wArticlesCleaned.0.json') as json_data:
	d = json.load(json_data)

with open('nytArticles.0.json') as json_data:
	newss = json.load(json_data)
print len(d),len(newss)

counts_wiki={}
counts_news={}
series=np.random.choice(len(d),13000)
for i in series:
	wiki=d[i]['text']
	words=wiki.split()
	for w in words:
		remove_punctuation(w)
		w=w.lower()
		if w not in counts_wiki:
			counts_wiki[w]=0
		counts_wiki[w]+=1  

for i in range(len(newss)):
	article=newss[i]["scrapedText"]
	words=article.split()
	for w in words:
		remove_punctuation(w)
		w=w.lower()
		if w not in counts_news:
			counts_news[w]=0
		counts_news[w]+=1
def formatting(w):
	l=list(w)
	if l[0]=="'":
		l=l[2:]
	word=''.join(l)
	word=word.encode('ascii','ignore')
	word = word.translate(None, ",.'[]:?!;()=*|-\"#%")
	return word

newswords=[]
wikiwords=[]

for k in counts_news:
	if counts_news[k]>=1000*counts_wiki.get(k,0) and counts_news[k]>100:
		l=list(k)
		if l[0]=="'":
			l=l[2:]
		word=''.join(l)
		word=word.encode('ascii','ignore')
		word = word.translate(None, ",.'[]:?!;()")
		if word and word not in newswords:
			newswords.append(word)
			
for k in counts_wiki:
	if counts_wiki[k]>=500*counts_news.get(k,0) and counts_wiki[k]>1000:
		word=formatting(k)
		if word and word not in wikiwords and word.isalpha():
			wikiwords.append(word)
					   
print newswords
print len(newswords)
print("------------")
print wikiwords
print len(wikiwords)
featurewords=newswords+wikiwords
print len(featurewords)


#is number, ?, !, %
data=[]
K=len(featurewords)

for i in range(1000):
	wiki=d[i]['text']
	words=wiki.split()
	l=np.zeros(K+4)
	for w in words:
		if "?" in list(w) or "!" in list(w):
			l[K+1]=1
		if "%" in list(w):
			l[K+2]=1
		remove_punctuation(w)
		w=w.lower()
		w=formatting(w)
		if w in featurewords:
			index=featurewords.index(w)
			l[index]+=1
		if w.isdigit():
			l[K]=1
		l[K+3]=1    #wikipedia
	data.append(l)
	
for i in range(1000):
	news=newss[i]["scrapedText"]
	words=news.split()
	l=np.zeros(K+4)
	for w in words:
		if "?" in list(w) or "!" in list(w):
			l[K+1]=1
		if "%" in list(w):
			l[K+2]=1
		remove_punctuation(w)
		w=w.lower()
		w=formatting(w)
		if w in featurewords:
			index=featurewords.index(w)
			l[index]+=1
		if w.isdigit():
			l[K]=1
		l[K+3]=0   #news
	data.append(l)
	
data=np.array(data)
print data.shape

tau=8.
def svm_train(matrix, category):
    state = {}
    M, N = matrix.shape
    #####################
    Y = category
    matrix = 1. * (matrix > 0)
    squared = np.sum(matrix * matrix, axis=1)
    gram = matrix.dot(matrix.T)
    K = np.exp(-(squared.reshape((1, -1)) + squared.reshape((-1, 1)) - 2 * gram) / (2 * (tau ** 2)) )

    alpha = np.zeros(M)
    alpha_avg = np.zeros(M)
    L = 1. / (64 * M)
    outer_loops = 40

    alpha_avg
    for ii in xrange(outer_loops * M):
        i = int(np.random.rand() * M)
        margin = Y[i] * np.dot(K[i, :], alpha)
        grad = M * L * K[:, i] * alpha[i]
        if (margin < 1):
            grad -=  Y[i] * K[:, i]
        alpha -=  grad / np.sqrt(ii + 1)
        alpha_avg += alpha

    alpha_avg /= (ii + 1) * M

    state['alpha'] = alpha
    state['alpha_avg'] = alpha_avg
    state['Xtrain'] = matrix
    state['Sqtrain'] = squared
    ####################
    return state

def svm_test(matrix, state):
    M, N = matrix.shape
    output = np.zeros(M)
    ###################
    Xtrain = state['Xtrain']
    Sqtrain = state['Sqtrain']
    matrix = 1. * (matrix > 0)
    squared = np.sum(matrix * matrix, axis=1)
    gram = matrix.dot(Xtrain.T)
    K = np.exp(-(squared.reshape((-1, 1)) + Sqtrain.reshape((1, -1)) - 2 * gram) / (2 * (tau ** 2)))
    alpha_avg = state['alpha_avg']
    preds = K.dot(alpha_avg)
    output = np.sign(preds)
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print 'Error: %1.4f' % error
    return error


np.random.seed(100)
p = np.random.permutation(data.shape[0])
data = data[p,:]
labels=data[:,-1]
data=data[:,range(data.shape[1]-1)]
cutoff=int(math.floor(3.0/10*data.shape[0]))
testData = data[range(cutoff),:]
print testData.shape[0],'test size'
testLabels = labels[range(cutoff)]
trainData = data[cutoff:,:]
print trainData.shape[0],'training size'
trainLabels = labels[cutoff:]

print 'start SVM training'
state = svm_train(trainData, trainLabels)
output = svm_test(testData, state)
evaluate(output, testLabels)

print("--- %s seconds ---" % (time.time() - start_time))

