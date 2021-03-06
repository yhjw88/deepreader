from tfidf import TFIDF
import util

#tfidf relevancy function
def reward2(s1,s2):
    indices = corpus[:]
    tfi=TFIDF()
    tfidf = tfi.get_tfidf(corpus)
    score = tfi.relevancy(tfidf, indices, s1,s2)
    return score+1


#a very simple baseline relevancy model, we need more sofisticated ones!!
def reward(s1,s2):
    l1=list(s1)
    l2=list(s2)
    return len(list(set(l1).intersection(l2)))+1

 #UCS search for our insertion problem!
class InsertionProblem(util.SearchProblem):
    def __init__(self, sentences, paragraph, reward):
        self.sentences=sentences
        self.para=paragraph
        self.reward = reward

    def startState(self):
        return (self.para[0],self.para[1],tuple(self.sentences),0)

    def isEnd(self, state):
        return str(state[1])==str(self.para[-1]) 

    def succAndCost(self, state):
        succs=[]
        for i in range(len(state[2])):
            if state[2][i]==None:
                if state[3]+2<len(self.para):
                    nextstate=(self.para[state[3]+1],self.para[state[3]+2],state[2],state[3]+1)
                    r=1.0/(2*reward2(state[0],state[1]))
                    succs.append((None,nextstate,r))
            else:
                choices=list(state[2][:])
                del choices[i]
                nextstate=(state[2][i],self.para[state[3]+1],tuple(choices),state[3]+1)
                r=1.0/(reward2(state[0],state[2][i])+reward2(state[1],state[2][i]))
                succs.append((state[2][i],nextstate,r))
        return succs

def InsertSentences(sentences, para, reward):
    if len(para) == 0:
        return ''
    ucs = util.UniformCostSearch(verbose=0)
    sentences.append(None)
    ucs.solve(InsertionProblem(sentences, para, reward))
    newpara=[]
    start=0
    newpara.append(str(para[0]))
    for e in ucs.actions:
        if e==None:
            start+=1
            newpara.append(para[start])
        else:
            newpara.append(e)
    lastindex=para.index(newpara[-1])
    if lastindex!=len(para)-1:
        newpara=newpara+para[lastindex+1:]
    #return '. '.join(str(v) for v in newpara)
    return [str(v) for v in newpara]

if __name__ == "__main__":
    f = open('data/para1.txt')
    para = f.read().strip(' ').split('. ')
    f2=open('data/sentences.txt')
    sentences=f2.read().strip(' ').split('. ')
    #print para,'paragraph'
    #print len(sentences),'sentences'

    corpus=para+sentences

    print '\n'.join(InsertSentences(sentences,para,reward))
