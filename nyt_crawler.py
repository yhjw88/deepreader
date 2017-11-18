import requests
from bs4 import BeautifulSoup
session = requests.Session()
url = "http://www.nytimes.com/2004/12/20/health/how-about-not-curing-us-some-autistics-are-pleading.html"
req = session.get(url)
soup = BeautifulSoup(req.text)
paragraphs = soup.find_all('p', class_='story-body-text story-content')
for p in paragraphs:
    print p.get_text()