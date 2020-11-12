import pandas as pd 
import numpy as np 
import os 
from bs4 import BeautifulSoup
import googlemaps



#os.chdir('~/Downloads')
df = pd.read_csv('tokyo.csv')   # don't hard code this - make function to read in all csv files in directory

def clean_data(folder=".", dataname="tokyo_main.csv"):
	
	files = [f for f in os.listdir(folder) if os.path.isfile(f)]
	
	merged = []
	
	for f in files:
		filename, ext = os.path.splitext(f)
		if ext == '.csv':
			read = pd.read_csv(f)
			merged.appen(read)
	result = pd.concat(merged, axis = 0)


	# now clean result

	# make nearest eki variable that is just in time 
	result.near_eki_time = result.nearest_eki.str.split('歩').str[1]
	result.near_eki_time = result.near_eki_time.astype(float)

	# remove nearest eki extra text 
	result.nearest_eki = result.nearest_eki.str.split("駅").str[0]
	result.nearest_eki = result.nearest_eki.astype(str)

	# fix ku_name 
	result.ku_name = result.ku_name.str.split('東京都').str[1]
	result.ku_name = result.ku_name.str.split('区').str[0]
	result.ku_name = result.ku_name.astype(str)

	# get rid of m2 in square meters and convert it to a float 
	result.sqr_m = result.sqr_m.str.split('m2').str[0]
	result.sqr_m = result.sqr_m.astype(float)


	# get rid of /n in year_built
	result.year_built = result.year_built.str.replace('\n', '')
	result.year_built = result.year_built.astype(str)


	# convert apartment_type to string
	result.apartment_type = result.apartment_type.astype(str)



	# fix money parts
	# needs to be a for loop for if they have a "." symbol then add 00's afterwards, else 000's 
	if "." in df.rent_price:
	    df.rent_price = df.rent_price.str.replace('万円', '000')
	elif ".##" in df.rent_price:
	    df.rent_price = df.rent_price.str.replace('万円', '00')
	    
	else:
	    df.rent_price = df.rent_price.str.replace('万円', '000')
	df.rent_price = df.rent_price.str.replace('.', '')
	df.rent_price = df.rent_price.astype(float)

	# skikikin price 

	if "." in df.rent_price:
	    df.rent_price = df.rent_price.str.replace('万円', '000')
	elif ".##" in df.rent_price:
	    df.rent_price = df.rent_price.str.replace('万円', '00')
	    
	else:
	    df.rent_price = df.rent_price.str.replace('万円', '000')
	df.rent_price = df.rent_price.str.replace('.', '')
	df.rent_price = df.rent_price.astype(float)

	# rei price 
	if "." in df.rent_price:
	    df.rent_price = df.rent_price.str.replace('万円', '000')
	elif ".##" in df.rent_price:
	    df.rent_price = df.rent_price.str.replace('万円', '00')
	    
	else:
	    df.rent_price = df.rent_price.str.replace('万円', '000')
	df.rent_price = df.rent_price.str.replace('.', '')
	df.rent_price = df.rent_price.astype(float)

	# fix maintainence price - remove yen symbol and make float 
	result.maintence_price = result.maintence_price.str.split('円').str[0]
	result.maintence_price = result.maintence_price.astype(float)

	# convert name of place into string 
	result.name_place = result.name_place.astype(str)

	# convert house type into string 
	result.house_type = result.house_type.astype(str)


	# convert 



	result.to_csv(dataname)











# make nearest eki variable that is just in time 
df.near_eki_time = df.nearest_eki.str.split('歩').str[1]

# remove yen symbol from monetary stuff
#df.rent_price = df.rent_price.str.replace('万円', '')

#df.rei_price = df.rei_price.str.replace('万円', '')

#df.maintence_price = df.maintence_price.str.replace('万円', '')

# convert numeric values into proper numeric values etc. 



#gmaps = googlemaps.Client(key="")

# add lon and lat variables in dataset 
for i in range(len(df)):
		geocode_ = gmaps.geocode(df.loc[i, 'address'])
	try:
		lat = geocode_[0]['geometry']['location']['lat']
		lng = geocode_[0]['geometry']['location']['lng']
		df_.loc[1,'Lat'] = lat
		df_.loc[1,'Lon'] = lng 
	except:
		lat = None
		lng = None