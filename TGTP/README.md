

# TGTP: Lightweight Traffic Poisoning Attack against NIDS 

 <p align="center">
  <a href='https://github.com/ZGC-LLM-Safety/TrafficLLM'><img src='https://img.shields.io/badge/Project-Page-purple'></a> 
  <a href='https://drive.google.com/drive/folders/1RZAOPcNKq73-quA8KG_lkAo_EqlwhlQb'><img src='https://img.shields.io/badge/Datasets-Page-red'></a> 
  <a href='https://github.com/ZGC-LLM-Safety/TrafficLLM/tree/master/models'><img src='https://img.shields.io/badge/Models-Page-green'></a> 
  <a href='https://mp.weixin.qq.com/s/pt2CfG0i9Fex-sy7-dcoyg' target='_blank'><img src='https://img.shields.io/badge/Chinese-Blog-blue'></a>
</p>

## **Overview** 

This repository contains the official implementation of our paper **"TGTP: Lightweight Traffic Poisoning Attack against NIDS"**, which proposes a novel framework for targeted clean - label poisoning attacks on Network Intrusion Detection Systems (NIDS). TGTP achieves high attack success rates with minimal perturbations and computational overhead. 

 ## 🔑**Key Contributions** 

1. **Two - Stage Guidance Strategy**:     
   - **Intra - class Sparse Space Guidance**: Uses p - value to identify outliers and disperse samples into sparse regions within the feature space.    
   - **Inter - class Distance Reduction**: Leverages SHAP values to minimize feature differences between benign and malicious traffic.   
2. **Hybrid Optimization**:    
   - **Continuous Features**: Gradient alignment for efficient perturbation optimization. 
   - **Discrete Features**: Metaheuristic algorithm (HO algorithm) for discrete space search.   
3. **Performance**:    
   - **Attack Success Rate**: 8.75% higher than SOTA at 0.05% poisoning budget.    
   - **Time Efficiency**: Only 27.2% of the computational cost compared to SOTA methods.  
 ## **Code Structure**

### Code Structure Summary

The codebase is organized into several modules, each with specific responsibilities related to data poisoning attacks in the context of machine learning models. Here is a detailed breakdown of the code structure:



```
TGTP/
├── README.md
├── Inter-class.py
├── train_model.py
├── attack.py
├── requirements.txt
├── forest/
│   ├── victims/
│   │   ├── training.py
│   │   ├── optimization_strategy.py
│   │   ├── victim_single.py
│   │   ├── victim_base.py
│   │   ├── victim_ensemble.py
│   │   ├── mobilenet.py
│   │   ├── models.py
│   ├── witchcoven/
│   │   ├── __init__.py
│   │   ├── witch_base.py
│   │   ├── witch_bullseye.py
│   │   ├── witch_watermark.py
│   ├── consts.py
│   ├── options.py
│   ├── utils.py
│   ├── gcloud.py
```



## 📥 **Installation Guide**

### 1. Clone the repository:

```
git clone https://github.com/HHLinlife/TGTP.git
cd TGTP
```

### 2. Install dependencies:

```
pip install -r requirements.txt
```



## 🛠 **Usage Instructions**

**1. Train a Surrogate Model**:

```
python train_model.py --model Var-CNN --dataset CIC-DDoS2019
```

**2. Generate Poisoned Samples**:

```
python attack.py --net Var-CNN --eps 8 --budget 0.001 --dataset ISCXTor2016 --data_path Traffic --pbatch 128
```



```
root@autodl-container-0b184db1e3-a1ea3321:~/autodl-fs/poisoning-gradient-matching-master# python attack.py --net Var-CNN --eps 8 --budget 0.001 --dataset ISCXTor2016 --data_path Traffic --pbatch 128
Currently evaluating -------------------------------:
Thursday, 10. October 2024 08:20AM

Starting clean training ...
Epoch: 0  | lr: 0.1000 | Training    loss is  2.0188, train acc:  29.55% | Validation   loss is  1.6065, valid acc:  42.46% | 
Epoch: 0  | lr: 0.1000 | Target adv. loss is  2.5581, fool  acc:   0.00% | Target orig. loss is  1.0427, orig. acc: 100.00% | 
Epoch: 1  | lr: 0.1000 | Training    loss is  1.4742, train acc:  46.01% | 
Epoch: 2  | lr: 0.1000 | Training    loss is  1.2024, train acc:  56.85% | 
Epoch: 3  | lr: 0.1000 | Training    loss is  0.9523, train acc:  66.42% | 
Epoch: 4  | lr: 0.1000 | Training    loss is  0.7649, train acc:  73.34% | 
Epoch: 5  | lr: 0.1000 | Training    loss is  0.6644, train acc:  76.98% | 
Epoch: 6  | lr: 0.1000 | Training    loss is  0.5966, train acc:  79.60% | 
Epoch: 7  | lr: 0.1000 | Training    loss is  0.5507, train acc:  81.10% | 
Epoch: 8  | lr: 0.1000 | Training    loss is  0.5189, train acc:  82.26% | 
Epoch: 9  | lr: 0.1000 | Training    loss is  0.4938, train acc:  83.05% | 
Epoch: 10 | lr: 0.1000 | Training    loss is  0.4758, train acc:  83.76% | Validation   loss is  0.5213, valid acc:  82.07% | 
Epoch: 10 | lr: 0.1000 | Target adv. loss is  5.6567, fool  acc:   0.00% | Target orig. loss is  0.1986, orig. acc: 100.00% | 
Epoch: 11 | lr: 0.1000 | Training    loss is  0.4545, train acc:  84.45% | 
Epoch: 12 | lr: 0.1000 | Training    loss is  0.4455, train acc:  84.71% | 
Epoch: 13 | lr: 0.0100 | Training    loss is  0.4331, train acc:  85.24% | 
Epoch: 14 | lr: 0.0100 | Training    loss is  0.2663, train acc:  90.92% | 
Epoch: 15 | lr: 0.0100 | Training    loss is  0.2135, train acc:  92.72% | 
Epoch: 16 | lr: 0.0100 | Training    loss is  0.1895, train acc:  93.54% | 
Epoch: 17 | lr: 0.0100 | Training    loss is  0.1760, train acc:  93.91% | 
Epoch: 18 | lr: 0.0100 | Training    loss is  0.1589, train acc:  94.55% | 
Epoch: 19 | lr: 0.0100 | Training    loss is  0.1474, train acc:  94.90% | 
Epoch: 20 | lr: 0.0100 | Training    loss is  0.1387, train acc:  95.26% | Validation   loss is  0.2699, valid acc:  91.47% | 
Epoch: 20 | lr: 0.0100 | Target adv. loss is  6.0076, fool  acc:   0.00% | Target orig. loss is  0.7265, orig. acc:   0.00% | 
Epoch: 21 | lr: 0.0100 | Training    loss is  0.1295, train acc:  95.43% | 
Epoch: 22 | lr: 0.0100 | Training    loss is  0.1171, train acc:  96.06% | 
Epoch: 23 | lr: 0.0010 | Training    loss is  0.1127, train acc:  95.92% | 
Epoch: 24 | lr: 0.0010 | Training    loss is  0.0816, train acc:  97.35% | 
Epoch: 25 | lr: 0.0010 | Training    loss is  0.0680, train acc:  97.73% | 
Epoch: 26 | lr: 0.0010 | Training    loss is  0.0630, train acc:  97.94% | 
Epoch: 27 | lr: 0.0010 | Training    loss is  0.0591, train acc:  98.10% | 
Epoch: 28 | lr: 0.0010 | Training    loss is  0.0543, train acc:  98.34% | 
Epoch: 29 | lr: 0.0010 | Training    loss is  0.0516, train acc:  98.32% | 
Epoch: 30 | lr: 0.0010 | Training    loss is  0.0490, train acc:  98.44% | Validation   loss is  0.2600, valid acc:  92.52% | 
Epoch: 30 | lr: 0.0010 | Target adv. loss is  9.9639, fool  acc:   0.00% | Target orig. loss is  1.0867, orig. acc:   0.00% | 
Epoch: 31 | lr: 0.0010 | Training    loss is  0.0463, train acc:  98.55% | 
Epoch: 32 | lr: 0.0010 | Training    loss is  0.0460, train acc:  98.50% | 
Epoch: 33 | lr: 0.0010 | Training    loss is  0.0428, train acc:  98.63% | 
Epoch: 34 | lr: 0.0001 | Training    loss is  0.0400, train acc:  98.70% | 
Epoch: 35 | lr: 0.0001 | Training    loss is  0.0370, train acc:  98.85% | 
Epoch: 36 | lr: 0.0001 | Training    loss is  0.0375, train acc:  98.82% | 
Epoch: 37 | lr: 0.0001 | Training    loss is  0.0367, train acc:  98.85% | 
Epoch: 38 | lr: 0.0001 | Training    loss is  0.0341, train acc:  98.94% | 
Epoch: 39 | lr: 0.0001 | Training    loss is  0.0358, train acc:  98.87% | Validation   loss is  0.2665, valid acc:  92.66% | 
Epoch: 39 | lr: 0.0001 | Target adv. loss is 10.6909, fool  acc:   0.00% | Target orig. loss is  0.5242, orig. acc: 100.00% | 
Starting brewing procedure ...
Target Grad Norm is 201.46749877929688
Iteration 0: Target loss is 0.7858, Poison clean acc is 94.00%
Iteration 50: Target loss is 0.3445, Poison clean acc is 72.00%
Iteration 100: Target loss is 0.2882, Poison clean acc is 74.00%
Iteration 150: Target loss is 0.2976, Poison clean acc is 84.00%
Iteration 200: Target loss is 0.2975, Poison clean acc is 80.00%
Iteration 249: Target loss is 0.2843, Poison clean acc is 82.00%
Iteration 0: Target loss is 0.7074, Poison clean acc is 98.00%
Iteration 50: Target loss is 0.3103, Poison clean acc is 70.00%
Iteration 100: Target loss is 0.2825, Poison clean acc is 74.00%
Iteration 150: Target loss is 0.2982, Poison clean acc is 78.00%
Iteration 200: Target loss is 0.2833, Poison clean acc is 78.00%
Iteration 249: Target loss is 0.2772, Poison clean acc is 76.00%
Iteration 0: Target loss is 0.6923, Poison clean acc is 98.00%
Iteration 50: Target loss is 0.2881, Poison clean acc is 70.00%
Iteration 100: Target loss is 0.2867, Poison clean acc is 76.00%
Iteration 150: Target loss is 0.2734, Poison clean acc is 78.00%
Iteration 200: Target loss is 0.2818, Poison clean acc is 84.00%
Iteration 249: Target loss is 0.2930, Poison clean acc is 78.00%
Iteration 0: Target loss is 0.6952, Poison clean acc is 96.00%
Iteration 50: Target loss is 0.2829, Poison clean acc is 76.00%
Iteration 100: Target loss is 0.2828, Poison clean acc is 74.00%
Iteration 150: Target loss is 0.3199, Poison clean acc is 78.00%
Iteration 200: Target loss is 0.2942, Poison clean acc is 80.00%
Iteration 249: Target loss is 0.3195, Poison clean acc is 86.00%
Iteration 0: Target loss is 0.6095, Poison clean acc is 94.00%
Iteration 50: Target loss is 0.3159, Poison clean acc is 70.00%
Iteration 100: Target loss is 0.2883, Poison clean acc is 74.00%
Iteration 150: Target loss is 0.2977, Poison clean acc is 80.00%
Iteration 200: Target loss is 0.2963, Poison clean acc is 78.00%
Iteration 249: Target loss is 0.3140, Poison clean acc is 82.00%
Iteration 0: Target loss is 0.7974, Poison clean acc is 98.00%
Iteration 50: Target loss is 0.3231, Poison clean acc is 78.00%
Iteration 100: Target loss is 0.3085, Poison clean acc is 76.00%
Iteration 150: Target loss is 0.2830, Poison clean acc is 78.00%
Iteration 200: Target loss is 0.2637, Poison clean acc is 80.00%
Iteration 249: Target loss is 0.3052, Poison clean acc is 82.00%
Iteration 0: Target loss is 0.6652, Poison clean acc is 98.00%
Iteration 50: Target loss is 0.2966, Poison clean acc is 70.00%
Iteration 100: Target loss is 0.2860, Poison clean acc is 78.00%
Iteration 150: Target loss is 0.2843, Poison clean acc is 76.00%
Iteration 200: Target loss is 0.2777, Poison clean acc is 80.00%
Iteration 249: Target loss is 0.2994, Poison clean acc is 76.00%
Iteration 0: Target loss is 0.5905, Poison clean acc is 98.00%
Iteration 50: Target loss is 0.2961, Poison clean acc is 72.00%
Iteration 100: Target loss is 0.2893, Poison clean acc is 76.00%
Iteration 150: Target loss is 0.2619, Poison clean acc is 76.00%
Iteration 200: Target loss is 0.2845, Poison clean acc is 76.00%
Iteration 249: Target loss is 0.3073, Poison clean acc is 80.00%
Poisons with minimal target loss 2.7716e-01 selected.
Var-CNN model initialized with random key 3260212679.
Model reinitialized to random seed.
Epoch: 0  | lr: 0.1000 | Training    loss is  2.0495, train acc:  27.27% | Validation   loss is  1.5886, valid acc:  40.24% | 
Epoch: 0  | lr: 0.1000 | Target adv. loss is  2.1339, fool  acc:   0.00% | Target orig. loss is  1.2337, orig. acc: 100.00% | 
Epoch: 1  | lr: 0.1000 | Training    loss is  1.5107, train acc:  44.33% | 
Epoch: 2  | lr: 0.1000 | Training    loss is  1.2717, train acc:  54.03% | 
Epoch: 3  | lr: 0.1000 | Training    loss is  1.0454, train acc:  62.68% | 
Epoch: 4  | lr: 0.1000 | Training    loss is  0.8743, train acc:  69.19% | 
Epoch: 5  | lr: 0.1000 | Training    loss is  0.7280, train acc:  74.73% | 
Epoch: 6  | lr: 0.1000 | Training    loss is  0.6447, train acc:  77.53% | 
Epoch: 7  | lr: 0.1000 | Training    loss is  0.5930, train acc:  79.53% | 
Epoch: 8  | lr: 0.1000 | Training    loss is  0.5489, train acc:  81.12% | 
Epoch: 9  | lr: 0.1000 | Training    loss is  0.5163, train acc:  82.16% | 
Epoch: 10 | lr: 0.1000 | Training    loss is  0.5001, train acc:  82.93% | Validation   loss is  0.6712, valid acc:  77.61% | 
Epoch: 10 | lr: 0.1000 | Target adv. loss is  5.7660, fool  acc:   0.00% | Target orig. loss is  0.0766, orig. acc: 100.00% | 
Epoch: 11 | lr: 0.1000 | Training    loss is  0.4799, train acc:  83.55% | 
Epoch: 12 | lr: 0.1000 | Training    loss is  0.4605, train acc:  84.26% | 
Epoch: 13 | lr: 0.0100 | Training    loss is  0.4472, train acc:  84.64% | 
Epoch: 14 | lr: 0.0100 | Training    loss is  0.2898, train acc:  90.24% | 
Epoch: 15 | lr: 0.0100 | Training    loss is  0.2312, train acc:  92.20% | 
Epoch: 16 | lr: 0.0100 | Training    loss is  0.2061, train acc:  93.04% | 
Epoch: 17 | lr: 0.0100 | Training    loss is  0.1873, train acc:  93.70% | 
Epoch: 18 | lr: 0.0100 | Training    loss is  0.1738, train acc:  94.00% | 
Epoch: 19 | lr: 0.0100 | Training    loss is  0.1653, train acc:  94.33% | 
Epoch: 20 | lr: 0.0100 | Training    loss is  0.1495, train acc:  94.86% | Validation   loss is  0.3127, valid acc:  89.69% | 
Epoch: 20 | lr: 0.0100 | Target adv. loss is  6.2596, fool  acc:   0.00% | Target orig. loss is  1.3477, orig. acc:   0.00% | 
Epoch: 21 | lr: 0.0100 | Training    loss is  0.1396, train acc:  95.20% | 
Epoch: 22 | lr: 0.0100 | Training    loss is  0.1352, train acc:  95.27% | 
Epoch: 23 | lr: 0.0010 | Training    loss is  0.1243, train acc:  95.68% | 
Epoch: 24 | lr: 0.0010 | Training    loss is  0.0920, train acc:  96.94% | 
Epoch: 25 | lr: 0.0010 | Training    loss is  0.0806, train acc:  97.36% | 
Epoch: 26 | lr: 0.0010 | Training    loss is  0.0724, train acc:  97.66% | 
Epoch: 27 | lr: 0.0010 | Training    loss is  0.0697, train acc:  97.72% | 
Epoch: 28 | lr: 0.0010 | Training    loss is  0.0638, train acc:  97.95% | 
Epoch: 29 | lr: 0.0010 | Training    loss is  0.0601, train acc:  98.05% | 
Epoch: 30 | lr: 0.0010 | Training    loss is  0.0589, train acc:  98.12% | Validation   loss is  0.2836, valid acc:  91.74% | 
Epoch: 30 | lr: 0.0010 | Target adv. loss is  7.1911, fool  acc:   0.00% | Target orig. loss is  2.3982, orig. acc:   0.00% | 
Epoch: 31 | lr: 0.0010 | Training    loss is  0.0545, train acc:  98.18% | 
Epoch: 32 | lr: 0.0010 | Training    loss is  0.0525, train acc:  98.34% | 
Epoch: 33 | lr: 0.0010 | Training    loss is  0.0510, train acc:  98.39% | 
Epoch: 34 | lr: 0.0001 | Training    loss is  0.0480, train acc:  98.46% | 
Epoch: 35 | lr: 0.0001 | Training    loss is  0.0435, train acc:  98.66% | 
Epoch: 36 | lr: 0.0001 | Training    loss is  0.0437, train acc:  98.65% | 
Epoch: 37 | lr: 0.0001 | Training    loss is  0.0418, train acc:  98.68% | 
Epoch: 38 | lr: 0.0001 | Training    loss is  0.0444, train acc:  98.57% | 
Epoch: 39 | lr: 0.0001 | Training    loss is  0.0410, train acc:  98.74% | Validation   loss is  0.2925, valid acc:  91.76% | 
Epoch: 39 | lr: 0.0001 | Target adv. loss is  7.3446, fool  acc:   0.00% | Target orig. loss is  3.3531, orig. acc:   0.00% | 

-------------Job finished.-------------------------
```



## 📦 Datasets

| Public Dataset | Type        | Samples | Download                                                     |
| :------------- | ----------- | ------- | ------------------------------------------------------------ |
| ISCXTor2016    | Tor Traffic | 80K     | [Tor-nonTor dataset (ISCXTor2016)](https://www.unb.ca/cic/datasets/tor.html ) |
| CIC-DDoS2019   | DDoS        | 51M     | [DDoS evaluation dataset (CIC-DDoS2019)](https://www.unb.ca/cic/datasets/ddos-2019.html) |
| CICDarknet2020 | Darknet     | 50K     | [CIC-Darknet2020](https://www.unb.ca/cic/datasets/darknet2020.html) |
| IoT-23         | IoT Attacks | 700K    | [CIC IoT dataset 2023]( https://www.unb.ca/cic/datasets/iotdataset-2023.html ) |



## 📊 Experimental Results

### Attack Performance (TGTP vs GDP)

| Metric        | TGTP  | GDP  |
| ------------- | ----- | ---- |
| ASR (%)       | 38.75 | 30   |
| VA (%)92.06   | 92.06 | 91.5 |
| Time Cost (m) | 18.2  | 53.4 |



## 📧 Contact

For questions or feedback, contact:

- Zhonghang Sui: <harry_hang@foxmail.com>
- Project Repository: [HHLinlife/TGTP](https://github.com/HHLinlife/TGTP)
