#! /bin/bash
#pip install -U torch==1.4.0 numpy==1.18.1
#pip install -r requirements.txt
#Generate graph data and store in /data/Twitter
python ./Process/getGraphNpz.py
#Reproduce the experimental results.
CUDA_VISIBLE_DEVICES=0 python ./model/t3rd.py --data_path='The location of your data'
#end
