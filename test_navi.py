# navitime scrape script test 
import requests 
import pandas as pd 
from bs4 import BeautifulSoup
import mechanize 

URL = "https://www.navitime.co.jp/maps/routeSearch"

# or can input what I am looking for as my destination in this url - input at goal and start locations 
goal = "東京都港区六本木６丁目１０−1"

start_ex = "東京都中野区新井１丁目２５−4" 


#https://www.navitime.co.jp/maps/routeResult?start=%7B%22name%22%3A%22%E3%83%91%E3%83%BC%E3%82%AF%E3%83%8F%E3%82%A4%E3%82%A2%E3%83%83%E3%83%88%E6%9D%B1%E4%BA%AC%22%2C%22lat%22%3A35.685494%2C%22lon%22%3A139.690747%2C%22spot%22%3A%2201140-YD3628315%22%2C%22road-type%22%3A%22default%22%7D&goal=%7B%22name%22%3A%22%E6%9D%B1%E4%BA%AC%E3%82%BF%E3%83%AF%E3%83%BC%22%2C%22lat%22%3A35.658636%2C%22lon%22%3A139.745406%2C%22spot%22%3A%2202301-1300505%22%2C%22road-type%22%3A%22default%22%7D&start-time=2020-11-24T18%3A42%3A00&walk-route=distance&rough-estimate=co2.taxi&road-fare=free&walk-speed=standard&term=2880
# this may not work though because we need lon and lat already 

# so instead start from routeSearch url and make sure cars is turned off then do a requests.post 


s = requests.session()
headers = {
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36',
}
s.headers.update(headers)

test = requests.get(URL, headers=headers)
soup = BeautifulSoup(test.text, 'html.parser')

