# FairCDR

This is the PyTorch implementation for our paper **"FairCDR: Transferring Fairness and User Preferences for Cross-Domain Recommendation."**

## Datasets

The dataset used can be found at: [https://tianchi.aliyun.com/dataset/408](https://tianchi.aliyun.com/dataset/408).  
The data preprocessing method is described in detail here: [https://github.com/datawhalechina/torch-rechub/tree/main/examples/ranking/data/ali-ccp](https://github.com/datawhalechina/torch-rechub/tree/main/examples/ranking/data/ali-ccp).  
The dataset is divided into three domains (1, 2, 3) based on the "Context Features."

## Environments
* python==3.8
* pytorch>=1.10.0
* numpy>=1.17.2
* scipy>=1.6.0


## Running the Codes

To obtain user and item representations for the target and source domains, run the following command:  
```bash
python main.py --train_type="pretrain"

To perform fairness transfer in Cross-Domain Recommendation (CDR), use the following command:
```python main.py --train_type="train"
