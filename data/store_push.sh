#!/bin/bash

# create bucket for gcp storeage 
gsutil mb -p jbrazzy -c COLDLINE -l us-east1-b -b on gs://housing_bura


# create folder for tokyo 

# make it so anyone can access tokyo folder 
gsutil acl ch -u AllUsers:R gs://housing_bura/tokyo




# push to gcp storage 
gsutil cp finished.csv gs://housing_bura/tokyo