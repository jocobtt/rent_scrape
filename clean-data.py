import pandas as pd 
import numpy as np 
import os 
from bs4 import BeautifulSoup


#os.chdir('~/Downloads')
df = pd.read_csv('apartment.csv')


# remove nearest eki html text that was still there  
df.nearest_eki = df.nearest_eki.str.split('">').str[1]
df.nearest_eki = df.nearest_eki.str.split('</').str[0]

# fix ku_name having weird symbols and number of listing at end
df.ku_name = df.ku_name.str.split('(').str[0]
df.ku_name = df.ku_name.str[1:]

# get rid of m2 in square meters and convert it to a float 
df.sqr_m = df.sqr_m.str.split('m2').str[0]
df.sqr_m = df.sqr_m.astype(float)

# get rid of /n in year_built
df.year_built = df.year_built.str.replace('\n', '')
df.year_built = df.year_built.astype('str')

# remove yen symbol from monetary stuff
df.rent_price = df.rent_price.str.replace('万円', '')

df.rei_price = df.rei_price.str.replace('万円', '')

df.maintence_price = df.maintence_price.str.replace('万円', '')

# convert numeric values into proper numeric values etc. 

# convert apartment_type to string
df.apartment_type = df.apartment_type.astype(str)

# convert 