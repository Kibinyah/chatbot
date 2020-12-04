from textblob import TextBlob
import requests
from bs4 import BeautifulSoup


class WebSearcher:
    def __init__(self, search):
        self.search = search
        self.url = 'https://en.wikipedia.org/wiki/{0}'.format(self.search)

    def run(self):
        description = ""

        response = requests.get(self.url)
        #print(response.text)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        blob = TextBlob(paragraphs[0].get_text())
        description = blob
        print(description)

def webSearch(userText):
    words = userText.split()
    term = words[-1]
    url = 'https://en.wikipedia.org/wiki/{0}'.format(term)
    data = requests.get(url)
    soup = BeautifulSoup(data.text, 'html.parser')
    paragraphs = soup.find_all('p')
    blob = TextBlob(paragraphs[1].get_text())
    print(blob)


a = WebSearcher('diabetes')
a.run()
webSearch("cancer")
