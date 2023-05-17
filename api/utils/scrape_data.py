import datasets 
import pandas as pd
import numpy as np
import os 
import logging
from bs4 import BeautifulSoup
from time import sleep
from datetime import datetime
import requests
from dotenv import load_dotenv
from datasets import load_dataset
load_dotenv()

class ScrapeData:
    def __init__(self, url, wait_time_min=1, wait_time_max=5, pages=(0, 50), output_directory=None):
        self.url = url
        self.wait_time_min = wait_time_min
        self.wait_time_max = wait_time_max
        self.pages = np.arange(*pages)
        self.output_directory = output_directory or os.getcwd()
        self.headers = {
            'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'
        }

def scrape(self):
        """Scrape data from the URL and return a DataFrame."""
        # Initialize lists for storing scraped data
        rent_price, address, sqr_m, name_place, rei_price = [], [], [], [], []
        eki, shikikin, floor, apartment_type, maintenence_price = [], [], [], [], []
        house_type, year_built, ku_name = [], [], []

        for page_num in self.pages:
            response = requests.get(f"{self.url}&pn={page_num}", headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            apartments = soup.find_all('div', class_="cassetteitem")
            sleep(np.random.randint(self.wait_time_min, self.wait_time_max))

            for container in apartments:
                # Extract and append each data point (some cleaning and checks added)
                rent_price.append(container.tbody.ul.span.text)
                address.append(container.div.ul.li.text)
                sqr_m.append(container.find('span', class_="cassetteitem_menseki").text)
                name_place.append(container.div.li.text)
                rei_price.append(container.find('span', class_="cassetteitem_price--gratuity").text)
                shikikin.append(container.find('span', class_="cassetteitem_price--deposit").text)
                eki.append(container.find('div', class_='cassetteitem_detail-text').text if container.find('div', class_='cassetteitem_detail-text') else "-")
                house_type.append(container.find('span', class_='ui-pct ui-pct--util1').text)
                floor.append(container.find('li', class_='cassetteitem_detail-col3').div.text)
                ku_name.append(soup.find('div', class_='designateline-box-txt02').text if soup.find('div', class_='designateline-box-txt02') else "-")
                year_built.append(container.find('li', class_='cassetteitem_detail-col3').text[5:])
                apartment_type.append(container.find("span", class_="cassetteitem_madori").text)
                maintenence_price.append(container.find('span', class_="cassetteitem_price--administration").text)
                sleep(np.random.randint(self.wait_time_min, self.wait_time_max))
        
        data = pd.DataFrame({
            'rent_price': rent_price, 
            'address': address,
            'sqr_m': sqr_m,
            'name_place': name_place,
            'rei_price': rei_price,
            'maintenence_price': maintenence_price,
            'nearest_eki': eki,
            'shikikin': shikikin,
            'apartment_type': apartment_type,
            'age': year_built,
            'ku_name': ku_name,
            'floor': floor,
            'house_type': house_type
        })
        
        return data
    
def clean_data(self, data: pd.DataFrame):
    """Clean scraped data."""
    data['rent_price'] = data['rent_price'].apply(self.convert_yen_to_number)
    data['rei_price'] = data['rei_price'].apply(self.convert_yen_to_number)
    data['shikikin'] = data['shikikin'].apply(self.convert_yen_to_number)
    data['maintenence_price'] = data['maintenence_price'].apply(self.convert_yen_to_number)
    data['year_built'] = data['age'].apply(self.convert_year_built)
    data['floor'] = data['floor'].apply(self.floor_fix)
    data['nearest_eki_walk'] = data['nearest_eki'].apply(self.convert_nearest_eki)
    data['nearest_eki_name'] = data['nearest_eki'].apply(self.eki_name)
    
    data = data.drop(columns=['nearest_eki', 'Lon', 'Lat'], errors='ignore')
    return data

def upload_to_hf(self, data: pd.DataFrame, hf_path: str, train_test_split: float = 0.2):
    """Upload cleaned data to Huggingface."""
    # is there anything we need to do around 
    # partition into train and test
    train_data, test_data = train_test_split(data, test_size=train_test_split)
    # save to csv
    train_data.to_csv("train.csv", index=False)
    test_data.to_csv("test.csv", index=False)
    # upload to Huggingface
    data_files = {"train": "train.csv", "test": "test.csv"}
    dataset = load_dataset(hf_path, data_files = data_files)
    return dataset
    
@staticmethod
def convert_year_built(x):
    return int(x.replace('年', '').replace('築', '').replace('-', '0')) if x != '新築' else 0

@staticmethod
def convert_yen_to_number(x):
    return float(x.replace('万円', '').replace('円', '').replace('-', '0'))

@staticmethod
def convert_nearest_eki(x):
    return int(x.split('歩')[1].split('分')[0]) if '歩' in x else 0

@staticmethod
def eki_name(x):
    return x.split('歩')[0] if '歩' in x else x

@staticmethod
def floor_fix(x):
    return int(x.replace("階", "").replace("地上", "").replace("\n", ""))

# Example usage
if __name__ == "__main__":
    url = "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&..."
    scraper = ScrapeData(url, wait_time_min=1, wait_time_max=3, pages=(0, 50))
    data = scraper.scrape()
    cleaned_data = scraper.clean_data(data)
    public_url = scraper.upload_to_hf(cleaned_data, "jbrazzy/tokyo_rent")
    print("Data uploaded to:", public_url)
