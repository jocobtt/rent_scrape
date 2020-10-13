import requests
import pandas as pd 
from bs4 import BeautifulSoup
import numpy as np 
import os 
from time import sleep

# this has all of the ku's I want - so use this instead of multiple URLs
URL = "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&pc=50&smk=&po1=25&po2=99&shkr1=03&shkr2=03&shkr3=03&shkr4=03&sc=13101&sc=13102&sc=13103&sc=13104&sc=13105&sc=13113&sc=13106&sc=13107&sc=13108&sc=13118&sc=13109&sc=13110&sc=13111&sc=13112&sc=13114&sc=13115&sc=13116&ta=13&cb=0.0&ct=9999999&md=02&md=03&md=04&md=05&md=06&et=9999999&mb=0&mt=9999999&cn=9999999&tc=0400301&fw2="


########################################
####      final code solution       ####
########################################

# print off that it is starting the loop 
print("Starting Loop...")



rent_price = []
address = []
sqr_m = []
name_place = []
rei_price = []
eki = []
apartment_type = []
maintenence_price = []
house_type = [] 
year_built = []
ku_name = []
floor = []
shikikin = []


s = requests.session()
headers = {
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36',
}
s.headers.update(headers)

pages = np.arange(1,2,1) # first test out on 100 then go from there 
for page in pages:
	page = requests.get(URL + str(pages) + "&ref_=adv_nxt", headers=headers)
	soup = BeautifulSoup(page.text, 'html.parser')
	apartment_container = soup.find_all('div', class_ = "cassetteitem")
	sleep(np.random.randint(2,20)) # 2-20 seconds between each page
	for container in apartment_container:
		sleep(np.random.randint(2,3)) # wait 2-5 seconds between each apartment container for testing 
		renty = container.tbody.ul.span.text
		rent_price.append(renty)
		sleep(np.random.randint(2,3))
		addy = container.div.ul.li.text
		address.append(addy)
		shik = container.find('span', class_="cassetteitem_price cassetteitem_price--deposit").text
		shikikin.append(shik) 
		sqr = container.find('span', class_="cassetteitem_menseki").text 
		sqr_m.append(sqr)
		name = container.find('div', class_="cassetteitem_content-title").text
		name_place.append(name)
		sleep(np.random.randint(2,3))
		rei = container.find('span', class_="cassetteitem_price cassetteitem_price--gratuity").text # will have to clean this later
		rei_price.append(rei)
		eki_ = container.find('div', class_='cassetteitem_detail-text').text
		eki.append(eki_)
		house = container.find('span',class_='ui-pct ui-pct--util1').text 
		house_type.append(house)
		sleep(np.random.randint(2,3))
		year = container.find('li', class_='cassetteitem_detail-col3').div.text
		year_built.append(year)
		ku = container.div.ul.li.text
		ku_name.append(ku)
		sleep(np.random.randint(2,3))
		floo = container.find('li', class_='cassetteitem_detail-col3').text[5:] # will have to drop stuff from this 
		floor.append(floo)
		apart = container.find("span", class_="cassetteitem_madori").text # will have to clean this later
		apartment_type.append(apart)
		admin = container.find('span', class_="cassetteitem_price cassetteitem_price--administration").text
		maintenence_price.append(admin)

# print that loop has completed 
print("Loop finished.... Converting to csv now.")






# save to data frame
df_ = pd.DataFrame({'rent_price': rent_price, 
	'address' : address,
	'sqr_m' : sqr_m,
	'name_place' : name_place,
	'rei_price' : rei_price,
	'shikikin' : shikikin,
	'maintenence_price' : maintenence_price,
	'nearest_eki' : eki,
	'apartment_type' : apartment_type,
	'year_built' : year_built,
	'ku_name' : ku_name,
	'floor' : floor,
	'house_type': house_type})


# os.chdir('~/Downloads/') -- don't need this yet
# save as csv
df_.to_csv('apartment.csv', index=False)

print("All done! Go checkout apartment.csv file now and run model and data cleaning file!")
