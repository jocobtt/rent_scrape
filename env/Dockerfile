FROM ubuntu:18.04

RUN apt-get update && apt-get -y update --fix-missing

RUN apt-get install -y vim

RUN apt-get install -y build-essential python3-pip python3-dev --fix-missing


RUN mkdir -p src/scrape
WORKDIR src/scrape
COPY . . 

RUN pip3 install -r requirements.txt

RUN cd $PWD

#CMD ["python3", "bs4_rent.py"]

#CMD ["python3", "scrape.py"]
