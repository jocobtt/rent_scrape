FROM python:3.9.0-slim

RUN apt-get update && apt-get -y update

RUN apt-get install -y vim

RUN mkdir -p src/scrape
WORKDIR src/scrape
COPY . . 

#RUN pip3 install -r requirements.txt
# need to fix this - getting weird error about pip install not working 
RUN pip3 install pandas && pip3 install bs4 && pip3 install requests

