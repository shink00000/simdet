# train on gcp

## run VM instance
1. Menu -> Compute Engine -> VM instance
1. create instance -> Marketplace
1. search 'pytorch' and select 'Deep Learning VM'
1. start
1. Machine Family -> GPU
1. Enter other items

## prepare data and set environment
```bash
git clone https://github.com/shink00000/simdet.git
cd simdet

bash ./docs/train_on_gcp.sh
```