# transit 
import requests 
import pandas as pd
import numpy as np 
import os 
import googlemaps 

print("beginning googlemaps loop/function for time to get to work")
# get time of to get to work via transit from potential apartment's nearest station via the googlemaps python api https://github.com/googlemaps/google-maps-services-python
gmaps = googlemaps.Client(key=)
# should this instead be a function? 

# try this as a function - still need to figure out what is wrong with getting transit info - will probably have to instead get info from eki to work instead 
def get_distance_(station, work, mode, transit_mode, address):  # should I include time I am going to commute at? 
# also put in how many address to do at a time clause? 
	time_to_work = []
	work = '6 Chome-10-1 Roppongi, Minato City, Tokyo 106-0032, Japan'

	osaka_ex = '4-chome-1 Tsurumi, Tsurumi-ku, Osaka, 538-0053, Japan'

	osaka_work = '1 Chome-4-16 Dojimahama, Kita Ward, Osaka, 530-0004, Japan'

	osaka_dest = '3-chome-2 Nakanoshima, Kita-ku, Osaka, 530-0005, Japan'
	
	# will instead have to look it up from closest station instead of actual address maybe?
	example_address = "4-chome-8 Togoshi, Shinagawa City, Tokyo-to 142-0041, Japan"

	station_example = '4-chome-25 Nerima, Nerima City, Tokyo-to 176-0001, Japan'

	destination_add = '9 Chome-7-39 Akasaka, Minato City, Tokyo 107-0052, Japan'

	# test out function first 
	geocode_result = gmaps.geocode(address)
	geocode_result = geocode_result[0]['geometry']['bounds']['northeast']

	# look to see how limited I am with the API, may have to break this up 
	#if transit_mode == "transit":

# https://medium.com/how-to-use-google-distance-matrix-api-in-python/how-to-use-google-distance-matrix-api-in-python-ef9cd895303c
# https://documenter.getpostman.com/view/6375424/RznJkbB5
# https://github.com/MKuranowski/TokyoGTFS
# https://api.trip2.jp/
# https://norikae.jorudan.co.jp/openapi/
#https://tokyochallenge.odpt.org/en/

	if len(apartment) > 100000:
		for  in address:
			addy = gmaps.geocode(address)  # will have to check this whole thing 
			lat = addy[0]["geometry"]["location"]["lat"]
			lng = addy[0]['geometry']['location']['lon']
			now = datetime.now()
			origins = (lng, lat)
			distance_result = gmaps.directions(origins, 
				work, 
				mode = mode, 
				transit_routing_preference = "fewer_transfers",
				transit_mode = ["subway", "train", "bus"]
				)
			time_to_work = distance_result['rows'][0]['elements'][0]['distance']['text'] # filter from output  
			time_result = distance_result['rows'][0]['elements'][0]['duration']['text']

	else:
		print("too many apartment listings in this dataset to do at once! Need to break it up")


# https://github.com/Zeletochoy/navitime/blob/master/navitime/cli.py
# run this loop for getting lon and lat - can probably just run this before I compile all of my datasets or maybe after?  
df['Lat'] = None
df['Lon'] = None

for i in range(len(df)):
	geocode_ = gmaps.geocode(df.loc[i, 'address'])
	try: 
	lat = geocode_[0]['geometry']['location']['lat']
	lng = geocode_[0]['geometry']['location']['lng']
	df.loc[1,'Lat'] = lat
	df.loc[1,'Lon'] = lng 
except:
	lat = None
	lng = None

def get_bikes(home, work, mode, transit_mode, address):  # should I include time I am going to commute at? 
# also put in how many address to do at a time clause? 
	bike_to_work = []
	work = '6 Chome-10-1 Roppongi, Minato City, Tokyo 106-0032, Japan'
	
	# will instead have to look it up from closest station instead of actual address maybe?
	example_address = "4-chome-8 Togoshi, Shinagawa City, Tokyo-to 142-0041, Japan"

	# test out function first 
	geocode_result = gmaps.geocode(address[1])
	geocode_result

	# look to see how limited I am with the API, may have to break this up 
	#if transit_mode == "transit":



	if len(apartment) > 100000:
		for  in address:
			addy = gmaps.geocode(address)  # will have to check this whole thing 
			lat = geocode_[0]['geometry']['location']['lat']
			lng = geocode_[0]['geometry']['location']['lng']
			origins = (lat, lng)
			now = datetime.now()
			distance_result = gmaps.directions(station, 
				work, 
				mode = mode
				)
			time_to_work = distance_result[0]['legs'][0]['duration']['text']
			distance_to_work = distance_result[0]['legs'][0]['distance']['text']
			

	else:
		print("too many apartment listings in this dataset to do at once! Need to break it up")


# https://github.com/Zeletochoy/navitime/blob/master/navitime/cli.py



