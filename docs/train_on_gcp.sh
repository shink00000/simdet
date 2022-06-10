#!/bin/bash

gsutil cp gs://simdet-data/voc07+12.zip .
gsutil cp -r gs://simdet-data/assets/* ./assets/
unzip voc07+12.zip -d ./data
rm voc07+12.zip

pip install torchmetrics==0.7.3
pip install tensorboard==2.8.0
pip install pycocotools==2.0.4
pip install scipy==1.6.3
pip install -e .
