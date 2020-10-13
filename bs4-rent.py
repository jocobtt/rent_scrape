# https://erikrood.com/Python_References/web_scrape.html
#https://linuxhint.com/python-beautifulsoup-tutorial-for-beginners/
# https://gist.github.com/ibnesayeed/b4b069b8007f68be91a179a099e35d71
#https://realpython.com/beautiful-soup-web-scraper-python/

import requests
import pandas as pd 
from bs4 import BeautifulSoup
import numpy as np 
import os 
from time import sleep
import googlemaps
from datetime import datetime

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
house_type = [] # cont.span
year_built = []
ku_name = []

s = requests.session()
headers = {
    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36',
}
s.headers.update(headers)

pages = np.arange(1,100,1) # first test out on 100 then go from there 
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
		shik = container.find('span', class_="cassetteitem_price cassetteitem_price--deposit")
		shikikin.append(shik) 
		sqr = container.find('span', class_="cassetteitem_menseki").text 
		sqr_m.append(sqr)
		name = container.find('div', class_="cassetteitem_content-title")
		name_place.append(name)
		sleep(np.random.randint(2,3))
		rei = container.find('span', class_="cassetteitem_price cassetteitem_price--gratuity").text # will have to clean this later
		rei_price.append(rei)
		eki_ = container.find('div', class_='cassetteitem_detail-text')
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
	'maintenence_price' : maintenence_price,
	'nearest_eki' : eki,
	'apartment_type' : apartment_type,
	'year_built' : year_built,
	'ku_name' : ku_name,
	'floor' : floor,
	'house_type': house_type})

# make long and lat variables in the dataset 
df['Lat'] = None
df['Lon'] = None

for i in range(len(df)):
	geocode_ = gmaps.geocode(df.loc[i, 'address'])
	try 
	lat = geocode_[0]['geometry']['location']['lat']
	lng = geocode_[0]['geometry']['location']['lng']
	df.loc[1,'Lat'] = lat
	df.loc[1,'Lon'] = lng 
except:
	lat = None
	lng = None

df['bike_time'] = None 
df['work_time'] = None
df['kyoukai_time'] = None

for i in range(len(df)):
	geocode_ = gmaps.geocode(df.loc[i, 'address'])
	try 
	lat = geocode_[0]['geometry']['location']['lat']
	lng = geocode_[0]['geometry']['location']['lng']
	df.loc[1,'Lat'] = lat
	df.loc[1,'Lon'] = lng 
except:
	lat = None
	lng = None

# os.chdir('~/Downloads/') -- don't need this yet
# save as csv
df_.to_csv('apartment.csv')

print("All done! Go checkout apartment.csv file now and run model and data cleaning file!")



# making this all a function now 
def rent_get(sumo_url, scrape_from, scrape_to, interval, wait_time_min, wait_time_max, directory):
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
	house_type = [] # cont.span
	year_built = []
	ku_name = []
	time_to_work = []
	nearest_ward = []
	bike_to_work = []
	geo_cords = []

	s = requests.session()
	headers = {
	    'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
	    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36',
	}
	s.headers.update(headers)

	pages = np.arange(scrape_from,scrape_to,interval) # first test out on 100 then go from there 
	for page in pages:
		page = requests.get(sumo_url + str(pages) + "&ref_=adv_nxt", headers=headers)
		soup = BeautifulSoup(page.text, 'html.parser')
		apartment_container = soup.find_all('div', class_ = "cassetteitem")
		sleep(np.random.randint(wait_time_min,wait_time_max)) # 2-20 seconds between each page
		for container in apartment_container:
			sleep(np.random.randint(wait_time_min,wait_time_min + 1)) # wait 2-5 seconds between each apartment container for testing 
			renty = container.tbody.ul.span.text
			rent_price.append(renty)
			sleep(np.random.randint(wait_time_min,wait_time_min + 1))
			addy = container.div.ul.li.text
			address.append(addy)
			sqr = container.find('span', class_="cassetteitem_menseki").text 
			sqr_m.appen(sqr)
			name = container.div.li.text
			name_place.append(name)
			sleep(np.random.randint(2,3))
			rei = container.find('span', class_="cassetteitem_price cassetteitem_price--gratuity").text # will have to clean this later
			rei_price.append(rei)
			eki_ = container.find('div', class_='cassetteitem_detail-text')
			eki.apend(eki_)
			house = container.find('span',class_='ui-pct ui-pct--util1').text 
			house_type.append(house)
			sleep(np.random.randint(2,3))
			floo = container.find('li', class_='cassetteitem_detail-col3').div.text
			floor.append(floo)
			ku = soup.find('div', class_='designateline-box-txt02').text  
			ku_name.append(ku)
			sleep(np.random.randint(2,3))
			year = container.find('li', class_='cassetteitem_detail-col3').text[5:] # will have to drop stuff from this 
			year_built.append(year)
			aprt = container.find("span", class_="cassetteitem_madori").text # will have to clean this later
			apartment_type.append(apart)
			admin = container.find('span', class_="cassetteitem_price cassetteitem_price--administration").text
			maintenence_price.append(admin)

	# print that main loop is now finished 
	print("Main scraping loop finished. Now pulling geocode information")

	gmaps = googlemaps.Client(key="my-keyfile")

	if len(address) > 1000:
		for addresse in address:
		geo_code = gmaps.geocode(addresse)
		geo_code = geo_code[0]["geometry"]["location"]

	else:
		print('Too many addresses to pull') 

    # get distance from eki to work 
    if len(address) > 1000:
    	for addresse in address:
    		work_address = work = '6 Chome-10-1 Roppongi, Minato City, Tokyo 106-0032, Japan'
    		time_to_work = gmaps.directions(address, work_address, mode = "transit")
    		time_to_work = time_to_work['rows']['routes']

	# save to data frame
	df_ = pd.DataFrame({'rent_price': rent_price, 
		'address' : address,
		'sqr_m' : sqr_m,
		'name_place' : name_place,
		'rei_price' : rei_price,
		'maintenence_price' : maintenence_price,
		'nearest_eki' : eki,
		'apartment_type' : apartment_type,
		'year_built' : year_built,
		'ku_name' : ku_name,
		'place_name' : place_name,
		'floor' : floor,
		'house_type': house_type,
		'to_work': time_to_work,
		'nearest_ward': nearest_ward,
		'geo_code' : geo_code})

	# os.chdir(directory) -- don't need this yet
	# save as csv
	df_.to_csv('apartment.csv', index=False)

	print("All done! Go checkout apartment.csv file now and run model and data cleaning file!")



def geo_coord(address):
	geo_code = []
	if len(address) > 1000:
		for adresses in address:
			geo_code = gmaps.geocode(addresses)

			return geo_code
	else:
		print("need a smaller dataset to pull from")

def get_distance_(station, work, mode, transi_mode, address):  # should I include time I am going to commute at? 
# also put in how many address to do at a time clause? 
	time_to_work = []
	work = '6 Chome-10-1 Roppongi, Minato City, Tokyo 106-0032, Japan'
	
	# will instead have to look it up from closest station instead of actual address maybe?
	example_address = "4-chome-8 Togoshi, Shinagawa City, Tokyo-to 142-0041, Japan"

	# test out function first 
	geocode_result = gmaps.geocode(address[1])

	# look to see how limited I am with the API, may have to break this up 
	#if transit_mode == "transit":



	if len(apartment) > 100000:
		for  in address:
			addy = gmaps.geocode(address)  # will have to check this whole thing 
			now = datetime.now()
			distance_result = gmaps.distance_matrix(origins = station, 
				destination = work, 
				mode = mode, 
				transit_routing_preference = "fewer_transfers",
				transit_mode = ["subway", "train", "bus"]
				)
			time_to_work = distance_result[1] # filter from output  
			

	else:
		print("too many apartment listings in this dataset to do at once! Need to break it up")
