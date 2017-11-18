from nltk.corpus import stopwords
import numpy as np
import lda
import string
import csv

translator = str.maketrans('', '', string.punctuation)
vocab = {}
articles = []
stop = stopwords.words("english")

with open('articles.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for article in reader:
        if len(article) > 0:
            ca = article[0].translate(translator)
            articles.append(ca)
            for w in ca.split():
                if w not in vocab.keys():
                    vocab[w] = 1

vi = list(vocab.keys())
print("words:", len(vi))
print("total:", str(len(articles)))

with open('topics.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['article', 'topics'])
    writer.writeheader()
    cnt = 1
    for desc in articles:
        if cnt > 200: break
        if cnt % 200 == 0:
            print("count:", cnt)
        cnt += 1
        row = np.zeros((1, len(vi)), dtype=np.int)
        for w in desc.split():
            if w in stop:
                continue
            row[0][vi.index(w)] += 1
        mat = row
        print(mat.shape)
        model = lda.LDA(n_topics=3, n_iter=1500, random_state=1)
        model.fit(mat)  # model.fit_transform(X) is also available
        topic_word = model.topic_word_  # model.components_ also works
        n_top_words = 2
        topics = []
        for i, topic_dist in enumerate(topic_word):
            sub = []
            for elem in np.argsort(topic_dist).astype(int):
                sub.append(vi[elem])
            topics.append(sub[:-n_top_words:-1])
        topics = ','.join(map(str, topics)) 
        writer.writerow({'article': desc, 'topics':topics})