#!/bin/bash

gsutil cp gs://simss-data/voc07+12.zip .
gsutil cp -r gs://simss-data/assets/* ./assets/
unzip voc07+12.zip -d ./data
rm voc07+12.zip

pip install torchmetrics==0.7.3
pip install tensorboard==2.8.0
pip install -e .
