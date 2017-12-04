import util

class InsertionProblem(util.SearchProblem):
    def __init__(self, sentences, paragraph, cost):
        self.sentences=sentences
        self.para=paragraph
        self.cost = cost

    def startState(self):
        return (self.para[0], self.para[1], tuple(self.sentences), 0)

    def isEnd(self, state):
        return str(state[1])==str(self.para[-1]) 

    def succAndCost(self, state):
        succs=[]
        for i in range(len(state[2])):
            if state[2][i]==None:
                if state[3]+2<len(self.para):
                    nextstate=(self.para[state[3]+1],self.para[state[3]+2],state[2],state[3]+1)
                    r = self.cost(state[0], state[1], None)
                    succs.append((None,nextstate,r))
            else:
                choices=list(state[2][:])
                del choices[i]
                nextstate=(state[2][i],self.para[state[3]+1],tuple(choices),state[3]+1)
                r = self.cost(state[0], state[1], state[2][i])
                succs.append((state[2][i],nextstate,r))
        return succs

def InsertSentences(sentences, para, cost):
    if len(para) == 0:
        return ''
    ucs = util.UniformCostSearch(verbose=0)
    sentences.append(None)
    ucs.solve(InsertionProblem(sentences, para, cost))
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
    return [str(v) for v in newpara]
