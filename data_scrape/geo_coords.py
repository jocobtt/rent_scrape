import os 
import requests 
import googlemaps 

address = ["4-chome-8 Togoshi, Shinagawa City, Tokyo-to 142-0041, Japan", "2-chome-8 Togoshi, Shinagawa City, Tokyo-to 142-0041, Japan"]



def geo_coord(address):
	geo_code = []
	if len(address) > 1000:
		for adresses in address:
			geo_code = gmaps.geocode(addresses)

			return geo_code
	else:
		print("need a smaller dataset to pull from")
