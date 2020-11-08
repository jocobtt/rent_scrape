import requests 
import googlemaps 
import pandas 
import os 
import numpy as np 
# optamize the distance from ward to my place 

# test apartment 
apart_address = "4-chome-8 Togoshi, Shinagawa City, Tokyo-to 142-0041, Japan"

# make a list of local church addresses to loop through 
church addresses = ['1 Chome-8-14 Eharacho, Nakano City, Tokyo 165-0023, Japan', '1 Chome-7-7 Kichijoji Higashicho, Musashino, Tokyo 180-0002, Japan', 
	'28-8 Sakuragaokacho, Shibuya City, Tokyo 150-0031, Japan', '2 Chome-16-17 Mizonokuchi, Takatsu Ward, Kawasaki, Kanagawa 213-0001, Japan', 
	'2 Chome-25-11 Minamisenzoku, Ota City, Tokyo 145-0063, Japan']

osaka_church = ['1 Chome-24-14 Sekime, Joto Ward, Osaka, 536-0008, Japan', '1 Chome-11-8 Hannancho, Abeno Ward, Osaka, 545-0021, Japan', 
	'2 Chome-8-13 Arakawa, Higashiosaka, Osaka 577-0843, Japan', '36-8 Tsukaguchicho 3-chome, Amagasaki, Hyogo 661-0002, Japan']

gmaps = googlemaps.Client(key="my-keyfile")


def church_dist(apart_address, church_addresses, mode):
	dist = []
	#apart_address =
	for addy in church_addresses:
		dist = gmaps.distance_matrix(origins = apart_address, 
			destination = addy,
			mode = mode,
			transit_routing_preference = "fewer_transfers",
			transit_mode = ["subway", "train", "bus"])

		return dist[1] 

