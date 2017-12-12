import pymongo
from pymongo import MongoClient
import requests
from bs4 import BeautifulSoup
import sys

def getArticleFromCNN(url):
    web = requests.get(url)
    soup = BeautifulSoup(web.text, 'lxml')
    article = soup.find("div", {"id": "storytext"})
    return article.text

def getArticleFromNYT(url):
    session = requests.Session()
    try:
        req = session.get(url)
    except requests.exceptions.ConnectionError:
        return None, None
    except requests.exceptions.MissingSchema:
        return None, None
    except requests.exceptions.InvalidSchema:
        return None, None

    soup = BeautifulSoup(req.text,"lxml")
    if not soup.title:
        return None, None
    title = soup.title.string
    paragraphs = soup.find_all('p', class_='story-body-text story-content')
    article = []
    for p in paragraphs:
        article.append(p.get_text())
    article = ' '.join(article)
    return title, article

def isGoodReference(reference):
    return (
        "url" in reference and
        reference["url"].find("nytimes") >= 0 and
        reference["url"].find("pdf") < 0
    )

# TODO: Merge urls that are the same.
def scrapeArticles(inCollection, outCollection, numToScrape=float('inf')):
    lastId = -1
    if outCollection.count() != 0:
        lastId = outCollection.find().sort([("wikipediaId", pymongo.DESCENDING)]).limit(1)[0]["wikipediaId"]
        print "Starting from id greater than: {}".format(lastId)
        sys.stdout.flush()
    numScraped = 0
    for article in inCollection.find({"_id": {"$gt": lastId}}).sort([("_id", pymongo.ASCENDING)]):
        # Exit if reached article limit.
        if numScraped >= numToScrape:
            break

        # Get list of relevant references.
        for reference in article["references"]:
            if isGoodReference(reference):
                sys.stdout.flush()
                reference["wikipediaId"] = article["_id"]
                reference["scrapedTitle"], reference["scrapedText"] = getArticleFromNYT(reference["url"])
                if not reference["scrapedText"]:
                    continue
                outCollection.insert_one(reference)

                # Print progress.
                numScraped += 1
                if numScraped % 5 == 0:
                    print "Scraped {} articles so far...".format(numScraped)
                    sys.stdout.flush()

def mergeArticles(collection):
    articles = list(collection.find())
    articleCache = {}
    numArticles = 0
    for article in articles:
        articleCache[article["url"]] = article
        numArticles += 1

        # if len(article["wikipediaId"]) > 0:
        #     newList = []
        #     for item in article["wikipediaId"]:
        #         if isinstance(item, list):
        #             newList.append(item[0])
        #         else:
        #             newList.append(item)
        #     collection.update_one({"_id": article["_id"]}, {"$set": {"wikipediaId": newList}})

        # if article["url"] not in articleCache:
        #     articleCache[article["url"]] = article
        # else:
        #     prevArticle = articleCache[article["url"]]
        #     if article["wikipediaId"] not in prevArticle["wikipediaId"]:
        #         prevArticle["wikipediaId"].append(article["wikipediaId"])
        #         collection.update_one({"_id": prevArticle["_id"]}, {"$set": {"wikipediaId": prevArticle["wikipediaId"]}})
        #     collection.delete_one({"_id": article["_id"]})
    print numArticles
    print len(articleCache)

def main():
    client = MongoClient()
    inCollection = client.cs229.wArticlesCleaned
    outCollection = client.cs229.nytArticles
    # mergeArticles(outCollection)
    scrapeArticles(inCollection, outCollection, float('inf'))

if __name__ == "__main__":
    main()
