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
#play around here more
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
structure_words=['external','links','disambiguation','reference','further','reading','also','see','history']

for k in counts_news:
	#play around here more
	if counts_news[k]>=100*counts_wiki.get(k,0) and counts_news[k]>500:
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
		if word and word not in wikiwords and word.isalpha() and word not in structure_words:
			wikiwords.append(word)
					   
print newswords
print len(newswords)
print("------------")
print wikiwords
print len(wikiwords)
#featurewords=newswords+wikiwords
featurewords=newswords
featurewords.append('eg')
print len(featurewords)


#is number, ?, !, %
data=[]
K=len(featurewords)

for i in range(10000):
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
	
for i in range(10000):
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


def nb_train(matrix, category):
	state = {}
	N = matrix.shape[1]
	###################
	m=matrix.shape[0]
	for token in range(N):
		l=[]
		l.append((sum([matrix[i,token] if category[i]==0 else 0 for i in range(m)])\
	+1.0)/(sum([matrix[i,:].sum() if category[i]==0 else 0 for i in range(m)])+N))
		l.append((sum([matrix[i,token] if category[i]==1 else 0 for i in range(m)])\
	+1.0)/(sum([matrix[i,:].sum() if category[i]==1 else 0 for i in range(m)])+N))
		state[token]=l
	state['y']=float(sum([1 if category[i]==1 else 0 for i in range(m)]))/m
	###################
	return state

def nb_test(matrix, state):
	output = np.zeros(matrix.shape[0])
	###################
	for doc in range(matrix.shape[0]):
		p1,p0=(0,0)
		for token in range(matrix.shape[1]):
			p1+=matrix[doc,token]*np.log(state[token][1])
			p0+=matrix[doc,token]*np.log(state[token][0])
		p1+=np.log(state['y'])
		p0+=np.log(1-state['y'])
		output[doc]=1 if p1>p0 else 0
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

print 'start training'
state = nb_train(trainData, trainLabels)
output = nb_test(testData, state)
evaluate(output, testLabels)
importance=[]
for token in range(trainData.shape[1]):
	importance.append(np.log(state[token][1]/state[token][0]))
ind = np.argpartition(importance, -5)[-5:]
importantwords=[]
for index in ind:
	if index<K:
		importantwords.append(featurewords[index])
	if index==K:
		importantwords.append('is_number')
	if index==K+1:
		importantwords.append('contains_?_!')
	if index==K+2:
		importantwords.append('contains_%')
print importantwords
print("--- %s seconds ---" % (time.time() - start_time))

