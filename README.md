
# Heterogeneous and Dynamic Aware Spatio-Temporal Learning for Route-based Travel Time Estimation
This a framework called HDTTE to incorporate the heterogeneous and dynamic informa- tion of spatio-temporal traffic for effective travel time estimation.
## Environment
- python 3.7.4
- torch 1.2.0
- numpy 1.17.2
## Dataset
Step 1： Download the processed Wuhan dataset from [Baidu Yun](https:###) 

If needed, the origin dataset of PEMSD4 and PEMSD8 are available from [ASTGCN](https://github.com/Davidham3/ASTGCN).

Step 2: Put them into data directories.
## Train command
    # Train with Wuhan dataset
    python train.py --data=WUHAN --is_train
    
    # Train with Beijing dataset
    python train.py --data=BJ --is_train
    
