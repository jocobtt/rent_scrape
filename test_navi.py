# navitime scrape script test 
import requests 
import pandas as pd 
from bs4 import BeautifulSoup
import selenium 
from selenium.webdriver.common.keys import keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options


chrome_options = Options()  
chrome_options.add_argument("--headless") 


URL = "https://www.navitime.co.jp/maps/routeSearch"

# or can input what I am looking for as my destination in this url - input at goal and start locations 
goal = "東京都港区六本木６丁目１０−1"
goal_official = "六本木ヒルズ"

start_ex = "東京都中野区新井１丁目２５−4" 


driver = webdriver.Chrome("/Users/jabras/Downloads/chromedriver", chrome_options = chrome_options)

if __name__ == '__main__':
    driver.get(URL)
    # remove cars as option - still not able to get there 
	#car_part = driver.find_elements_by_xpath('//*[@id="route-search-detail-link__default"]/a').
    # input address and click on start location search button 
    location_search = driver.find_element_by_class_name('start-input').send_keys(start_ex)  # instead of start_ex just have it use loop addresses
    # click on search button 
    start_button = driver.find_element_by_class_name('//*[@id="route-search-start-box"]/div[1]/button').click()
    # click first option that pops up for an address 
	click_start_op = driver.find_element_by_xpath('//*[@id="route-search-start-box"]/div[2]/section/ul[1]/li[1]/a').click()
	# input address and click on goal location search button 
	goal_search = driver.find_element_by_class_name('goal-input').send_keys(goal_official)
	# click on search button for goal 
	goal_button = driver.find_element_by_xpath('//*[@id="route-search-goal-box"]/div[1]/button').click()
	# click first address 
	click_goal_op = driver.find_element_by_xpath('//*[@id="route-search-goal-box"]/div[2]/section/ul[1]/li[1]/a').click() 
	# click on main search button 
	search = driver.find_element_by_xpath('//*[@id="route-search-condition-content"]/div/div[7]/button') # this doesn't totally work yet... 
	# print current url 
	current_url = driver.current_url
	lon =
	lat = 
	# get second route so not the car option - will change once I fix things earlier for that part...
	second = driver.find_element_by_xpath('//*[@id="ui-id-3"]').click()
	# pull down price 
	price = driver.find_element_by_xpath('//*[@id="route-2"]/div[1]/div[3]/div[2]/div[2]/div/div').text
	# get commmute time 
	time = driver.find_element_by_xpath('//*[@id="route-2"]/div[1]/div[1]/span[1]').text
	# number of train changes 
	changes = driver.find_element_by_xpath('//*[@id="route-2"]/div[1]/div[3]/div[1]/div[2]/div/div').text
	# wait some time 
	WebDriverWait(10s) # check this syntax 
	# get bike route only ...
	click_bike = driver.find_element_by_xpath('//*[@id="route-search-types"]/label[4]/div').click()
	# get bike distance 
	bike_dist = driver.find_element_by_xpath('//*[@id="route-1"]/div[1]/div[2]/div[1]/div[2]/div/div').text
	# get calories spent 
	calories = driver.find_element_by_xpath('//*[@id="route-1"]/div[1]/div[2]/div[2]/div[2]/div/div').text


#https://www.navitime.co.jp/maps/routeResult?start=%7B%22name%22%3A%22%E3%83%91%E3%83%BC%E3%82%AF%E3%83%8F%E3%82%A4%E3%82%A2%E3%83%83%E3%83%88%E6%9D%B1%E4%BA%AC%22%2C%22lat%22%3A35.685494%2C%22lon%22%3A139.690747%2C%22spot%22%3A%2201140-YD3628315%22%2C%22road-type%22%3A%22default%22%7D&goal=%7B%22name%22%3A%22%E6%9D%B1%E4%BA%AC%E3%82%BF%E3%83%AF%E3%83%BC%22%2C%22lat%22%3A35.658636%2C%22lon%22%3A139.745406%2C%22spot%22%3A%2202301-1300505%22%2C%22road-type%22%3A%22default%22%7D&start-time=2020-11-24T18%3A42%3A00&walk-route=distance&rough-estimate=co2.taxi&road-fare=free&walk-speed=standard&term=2880
# this may not work though because we need lon and lat and we don't have that yet 

# so instead start from routeSearch url and make sure cars is turned off then use something like this - https://github.com/python-mechanize/mechanize
# https://github.com/python-mechanize/mechanize/blob/master/examples/hack21.py
# https://www.ibm.com/developerworks/linux/library/l-python-mechanize-beautiful-soup/

def sel_get_inf(URL ,chromedriver, start, goal, orig_data_set):
	URL = URL 


# pseudo code 
#for apartment in apartment_list:
# click <a data-ga-click="{&quot;action&quot;:&quot;ルート検索_TOP&quot;, &quot;label&quot;:&quot;詳細条件設定&quot;}">詳細条件設定 +</a> to get to part to remove car route 
# click <input type="checkbox" name="car" id="only.mix" value="only.mix" checked="" data-ga-click="{&quot;action&quot;:&quot;ルート検索_TOP&quot;, &quot;label&quot;:&quot;トータルナビ条件_自動車&quot;}"> to get rid of car 
# input <input class="start-input" type="text" placeholder="出発地を入力"> for start location
# click <button class="search-btn orange-btn" data-ga-click="{&quot;action&quot;:&quot;ルート検索_TOP&quot;, &quot;label&quot;:&quot;地点検索ボタン&quot;}">
#						<div class="img-icon_btn_search"></div>
#					</button>
# will need to confirm that the location is the one I want - pick the first one 
# input <input class="goal-input" type="text" placeholder="目的地を入力"> for goal location 
# click <button class="search-btn orange-btn" data-ga-click="{&quot;action&quot;:&quot;ルート検索_TOP&quot;, &quot;label&quot;:&quot;地点検索ボタン&quot;}">
#						<div class="img-icon_btn_search"></div>
#					</button>
# click <button class="orange-btn " data-ga-click="{&quot;action&quot;:&quot;ルート検索_TOP&quot;, &quot;label&quot;:&quot;ルート検索ボタン&quot;}">ルート検索</button> to search the location 
# pull time to get there - <span class="total-time">45分</span>
# can get list of steps to get there from this - <ul class="sections"> == $0

# lon & lat 
# pull from URL 
# get url from mechanize? 
# follow this - https://stackoverflow.com/questions/24785532/how-to-find-the-current-url-in-python-mechanize

# biking info! 
# click biking part - <div class="img-icon_tab_bicycle_pressed" data-ga-click="{&quot;action&quot;:&quot;ルート検索_TOP&quot;, &quot;label&quot;:&quot;自転車タブ&quot;}"></div>
# get time - <span class="total-time">48分</span>
# get distance - <div>9.4km</div>

