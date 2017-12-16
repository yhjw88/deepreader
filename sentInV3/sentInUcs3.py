import dateutil.parser
import util

class InsertionProblem(util.SearchProblem):
    def __init__(self, sentences, section, cost, costData):
        self.sentences = sentences
        self.section = section
        self.cost = cost
        self.costData = costData
        self.yearCache = {}

    # Not the best but workable, doesn't work for BC.
    def getYear(self, sentence):
        if sentence in self.yearCache:
            return self.yearCache[sentence]

        theYear = None
        try:
            theYear = dateutil.parser.parse(sentence, fuzzy=True).year
        except ValueError as e:
            pass
        except OverflowError as e:
            pass
        self.yearCache[sentence] = theYear
        return theYear

    # (sentence1, sentence2, None U sentencesToInsert, insertionIndex, lastSeenYear).
    # insertionIndex is 0 to len(section), and len(section) + 1 when done.
    def startState(self):
        return (None, self.section[0], tuple(self.sentences), 0, None)

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
                nextSeenYear = self.getYear(state[1]) if state[1] else None
                nextstate = (
                    state[1],
                    self.section[nextInsertionIndex] if nextInsertionIndex < len(self.section) else None,
                    state[2],
                    nextInsertionIndex,
                    nextSeenYear if nextSeenYear else state[4])

                cost = 0
                if state[0] is not None and state[1] is not None:
                    cost += self.cost(state[0], state[1], None, self.costData)

                succs.append((action, nextstate, cost))
            else:
                actionsCopy = list(state[2][:])
                actionsCopy.pop(actionIndex)
                nextSeenYear = self.getYear(action)
                nextstate = (
                    action,
                    state[1],
                    tuple(actionsCopy),
                    state[3],
                    nextSeenYear if nextSeenYear else state[4])

                cost = 0
                if state[0] is not None:
                    cost += self.cost(state[0], action, None, self.costData)
                if state[4] and nextSeenYear and nextSeenYear < state[4]:
                    cost += 0.05

                succs.append((action, nextstate, cost))
        return succs

def InsertSentences(sentences, section, cost, costData):
    if len(section) == 0:
        return ''

    actions = list(sentences)
    actions.append(None)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(InsertionProblem(actions, section, cost, costData))

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
