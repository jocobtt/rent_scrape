#!/bin/bash

# build docker file I created
docker build -t jabras/python_scrape .

# run and apply docker file I created - maybe just have it run in the background -- also need to include r part to this 
docker run -it --rm -p 8888:8888 jabras/python_scrape

# apply kubernetes manifests
kubectl apply -k kustomization.yaml
