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
from datetime import datetime
from google.cloud import storage
# load our environment variables
from dotenv import load_dotenv
load_dotenv()

URL = "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&pc=30&smk=&po1=25&po2=99&shkr1=03&shkr2=03&shkr3=03&shkr4=03&sc=13101&sc=13102&sc=13103&sc=13104&sc=13105&sc=13113&sc=13106&sc=13107&sc=13108&sc=13118&sc=13121&sc=13122&sc=13123&sc=13109&sc=13110&sc=13111&sc=13112&sc=13114&sc=13115&sc=13120&ta=13&cb=0.0&ct=9999999&md=01&md=02&md=03&md=04&md=05&md=06&md=07&et=9999999&mb=0&mt=9999999&cn=9999999&fw2="


########################################
####      final code solution       ####
########################################

# making this all a function now 
def rent_get(sumo_url, scrape_from, scrape_to, interval, wait_time_min, wait_time_max, directory, data_set):
	# print off that it is starting the loop 
	print("Starting Loop...")


	rent_price = []
	address = []
	sqr_m = []
	name_place = []
	rei_price = []
	eki = []
	shikikin = []
	floor = []
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
			shik = container.find('span', class_="cassetteitem_price cassetteitem_price--deposit").text
			shikikin.append(shik) 
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
			apartment_type.append(aprt)
			admin = container.find('span', class_="cassetteitem_price cassetteitem_price--administration").text
			maintenence_price.append(admin)

	# print that main loop is now finished 
	print("Main scraping loop finished. Now pulling geocode information")



	# save to data frame
	df_ = pd.DataFrame({'rent_price': rent_price, 
		'address' : address,
		'sqr_m' : sqr_m,
		'name_place' : name_place,
		'rei_price' : rei_price,
		'maintenence_price' : maintenence_price,
		'nearest_eki' : eki,
		'shikikin': shikikin,
		'apartment_type' : apartment_type,
		'age' : year_built,
		'ku_name' : ku_name,
		'floor' : floor,
		'house_type': house_type})
 

	os.chdir(directory)
	print("All done! Go checkout file now and run model and data cleaning file!")
	
	return df_.to_csv(data_set, index=False)

def convert_year_built(x):
    # first check if it's not a nan value
    if x == '-':
        x = x.replace('-', '0')
        x = int(x)
        return x
    elif "新築" in x:
        x = x.replace('新築', '0')
        x = int(x)
        return x
    else:
        x = x.replace('年', '')
        x = x.replace('築', '')
        x = int(x)
        return x
    
def convert_yen_to_number(x):
    # remove man and yen sign
    # first check if it's not a nan value
    if x == '-': 
        x = x.replace('-', '0')
        x = float(x)
        return x
    else:
        x = x.replace('万円', '') # how to handle non man yen values 
        x = x.replace('円', '')
        x = float(x)
        return x
    
def convert_nearest_eki(x):
    # first check if it's not a nan value
    if x == '-' or x == 'nan':
        x = x.replace('-', '0')
        x = x.replace('nan', '0')
        x = int(x)
        return x
    else:
        # split string up to "歩"
        x = x.split('歩')[1]
        # split string up to "分"
        x = x.split('分')[0]
        return int(x)

def eki_name(x):
    # first check if it's not a nan value
    if x == '-' or x == 'nan':
        x = x.replace('-', '0')
        x = x.replace('nan', '0')
        x = str(x)
        return x
    else:
        # split string up to "歩"
        x = x.split('歩')[0]
        return str(x)
    
def floor_fix(x):
    x = x.replace("\n", "")
    if "地上" in x:
        x = x.split("地上")[1]
        x = x.split('階')[0]
    else:
        x = x.split('階')[0]
    return x

def clean_data(data, destination):
	print("Cleaning data now...")
	# clean the data 
	df = pd.read_csv(data)
	df['maintenence_price'] = df['maintenence_price'].fillna(0)
	# replace rei price with 0 if it is nan
	df['rei_price'] = df['rei_price'].fillna(0)
	# replace shiki price with 0 if it is nan
	df['shikikin'] = df['shikikin'].fillna(0)
	# drop columns where square meters is nan
	df = df.dropna(subset=['sqr_m'])
	# replace \n with nothing in all columns 
	df = df.replace('\n','', regex=True)
	# make square meters a float & drop the m2
	df['sqr_m'] = df['sqr_m'].str.split('m').str[0]
	df['sqr_m'] = df['sqr_m'].astype(float)
	# drop yen sign and make rent price a float
	df["rent_price"] = df["rent_price"].apply(convert_yen_to_number)
	df["rei_price"] = df["rei_price"].apply(convert_yen_to_number)
	df["shikikin"] = df["shikikin"].apply(convert_yen_to_number)
	df["maintenence_price"] = df["maintenence_price"].apply(convert_yen_to_number)
	
	# make year built be set from 0 to now 
	# fix ku name
	df["ku_name"] = df["ku_name"].str.split('東京都').str[1]
	df['ku_name'] = df["ku_name"].str.split('区').str[0]
	
	# fix year built
	df['year_built'] = df['year_built'].apply(convert_year_built)
	# fix floor
	df['floor'] = df['floor'].apply(floor_fix)
	df['floor'] = df['floor'].fillna(0)
	# replace "" with 0
	df['floor'] = df['floor'].replace('', '0')
	# make floor an int
	df['floor'] = df['floor'].astype(int)
	# drop lon and lat
	df = df.drop(['Lon', 'Lat'], axis = 1)
	# convert nearest eki to minute walk and station name
	df['eki_walk'] = df['nearest_eki'].apply(convert_nearest_eki)
	df['eki_name'] = df['nearest_eki'].apply(eki_name)
	# drop nearest eki
	df = df.drop(['nearest_eki'], axis = 1)
	print("All clean now! Pushing to cloud storage")
	os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
	storage_client = storage.Client()
	bucket_name = destination
	bucket = storage_client.bucket(bucket_name)
	
	# todays date 
	today = datetime.date.today()
	bytes_to_upload = df.to_csv(index=False).encode('utf-8')

	# upload csv to google storage - also want argument for region
	blob = bucket.blob(f'{data}_{today}.csv')
	blob.upload_from_string(bytes_to_upload)
	return blob.public_url

if __name__ == "__main__":
	rent_get(URL, 0, 50, 3, 1, 5, "~/Users/jabras/rent_scrape/data/", "Tokyo.csv")
	clean_data("~/Users/jabras/rent_scrape/data/Tokyo.csv", "tokyo_rental_data")