import pandas as pd 
import numpy as np 
import os 
from bs4 import BeautifulSoup
import googlemaps



#os.chdir('~/Downloads')
df = pd.read_csv('tokyo.csv')   # don't hard code this - make function to read in all csv files in directory

# combine the datasets 
test = pd.concat([df, df_2], axis = 0)


# remove nearest eki html text that was still there  
df.nearest_eki = df.nearest_eki.str.split('">').str[1]
df.nearest_eki = df.nearest_eki.str.split('</').str[0]

# fix ku_name 
df.ku_name = df.ku_name.str.split('東京都').str[1]
df.ku_name = df.ku_name_test.str.split('区').str[0]



# get rid of m2 in square meters and convert it to a float 
#df.sqr_m = df.sqr_m.str.split('m2').str[0]
#df.sqr_m = df.sqr_m.astype(float)

# get rid of /n in year_built
df.year_built = df.year_built.str.replace('\n', '')
df.year_built = df.year_built.astype('str')

# make nearest eki variable that is just in time 
df.near_eki_time = df.nearest_eki.str.split('歩').str[1]

# remove yen symbol from monetary stuff
#df.rent_price = df.rent_price.str.replace('万円', '')

#df.rei_price = df.rei_price.str.replace('万円', '')

#df.maintence_price = df.maintence_price.str.replace('万円', '')

# convert numeric values into proper numeric values etc. 

# convert apartment_type to string
df.apartment_type = df.apartment_type.astype(str)


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