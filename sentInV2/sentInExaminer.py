import dateutil.parser
import json
import random
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import sys
import timex

def main():
    # Solve utf errors.
    reload(sys)
    sys.setdefaultencoding('utf8')

    # Load data.
    filename = "data/sentInDev.json"
    dataList = []
    with open(filename, 'rb') as inFile:
        dataList = json.load(inFile)
    dataList = dataList[:10]

    # Examine cosine similarity scores between sentences.
    outputs = []
    for data in dataList:
        section = data["prelimSection"]
        corpus = list(section)
        tf = TfidfVectorizer(analyzer='word',
                             ngram_range=(1,3),
                             min_df = 0,
                             stop_words = 'english',
                             decode_error='ignore')
        tfidfMatrix = tf.fit_transform(corpus)
        cosineSimMatrix = linear_kernel(tfidfMatrix, tfidfMatrix)

        svd = TruncatedSVD(10)
        lsa = make_pipeline(svd, Normalizer(copy=False))
        lsaMatrix = lsa.fit_transform(tfidfMatrix)
        cosineSimMatrix = linear_kernel(lsaMatrix, lsaMatrix)

        nums = []
        for i in range(len(section)-1):
            nums.append(cosineSimMatrix[i][i+1])
        outputs.append("TfIdf: {0:.4f}".format(sum(nums) / len(nums)))

        nums = []
        permute = range(len(section)-1)
        random.shuffle(permute)
        for i, j in zip(permute[:len(permute)-1], permute[1:]):
            nums.append(cosineSimMatrix[i][j])
        outputs[-1] = outputs[-1] + " Permute: {0:.4f}".format(sum(nums) / len(nums))
    
    for output in outputs:
        print output

if __name__ == '__main__':
    # main()
    print dateutil.parser.parse("I did nothing in 500 BC", fuzzy=True).year
