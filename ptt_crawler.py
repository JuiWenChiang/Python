import random, time
import requests
from bs4 import BeautifulSoup

url = 'https://www.ptt.cc/bbs/prozac/index.html'
headers = {'uer-agent':''}


# 抓網址
def get_all_href(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    results = soup.select("div.title")
    for item in results:
        a_item = item.select_one("a")
        title = item.text
        if a_item:
            print('https://www.ptt.cc' + a_item.get('href'))


# 抓幾頁
for page in range(1, 100):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    btn = soup.select('div.btn-group > a')
    up_page_href = btn[3]['href']
    next_page_url = 'https://www.ptt.cc' + up_page_href
    url = next_page_url
    get_all_href(url=url)
    time.sleep(random.randint(3,10))





