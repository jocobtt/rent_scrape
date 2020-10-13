FROM ubuntu:18.04

RUN apt-get update && apt-get -y update 

RUN apt-get install -y build-essential python3-pip python3-dev 


RUN mkdir -p src/scrape
WORKDIR src/scrape
COPY bs4-rent.py bs4-rent.py

#RUN pip3 install -r requirements.txt

RUN pip3 install pandas 

RUN pip3 install bs4 

RUN pip3 install requests


RUN cd $PWD

# run my bs4-rent script 
CMD ["python3", "run", "bs4-rent.py"]

# how to run a second script - ie to clean my dataset

