import util

class InsertionProblem(util.SearchProblem):
    def __init__(self, sentences, section, cost):
        self.sentences = sentences
        self.section = section
        self.cost = cost

    # (sentence1, sentence2, None U sentencesToInsert, insertionIndex).
    # insertionIndex is 0 to len(section), and len(section) + 1 when done.
    def startState(self):
        return (None, self.section[0], tuple(self.sentences), 0)

    def isEnd(self, state):
        return state[3] == len(self.section) + 1

    def succAndCost(self, state):
        succs=[]
        for actionIndex, action in enumerate(state[2]):
            # Force insertion.
            if state[1] is None and len(state[2]) > 1 and action is None:
                continue

            if action is None:
                nextInsertionIndex = state[3] + 1
                nextstate = (
                    state[1],
                    self.section[nextInsertionIndex] if nextInsertionIndex < len(self.section) else None,
                    state[2],
                    nextInsertionIndex)

                if state[0] is None or state[1] is None:
                    cost = 0
                else:
                    cost = self.cost(state[0], state[1], None)

                succs.append((action, nextstate, cost))
            else:
                actionsCopy = list(state[2][:])
                actionsCopy.pop(actionIndex)
                nextstate = (
                    action,
                    state[1],
                    tuple(actionsCopy),
                    state[3])

                if state[0] is None:
                    cost = 0
                else:
                    cost = self.cost(state[0], action, None)
                succs.append((action, nextstate, cost))
        return succs

def InsertSentences(sentences, section, cost):
    if len(section) == 0:
        return ''

    actions = list(sentences)
    actions.append(None)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(InsertionProblem(actions, section, cost))

    newSection = []
    insertionIndex = 0
    for action in ucs.actions:
        if action is None and insertionIndex >= len(section):
            break

        if action is None:
            newSection.append(section[insertionIndex])
            insertionIndex += 1
        else:
            newSection.append(action)

    return [str(v) for v in newSection]
