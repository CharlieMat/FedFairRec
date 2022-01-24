# FedFairRec

Federated Fairness-aware Recommendation

## 0. Setup

Code environment:

```
> conda create -n ffrec python=3.9
> conda activate ffrec
> conda install -c anaconda ipykernel
> python -m ipykernel install --user --name ffrec --display-name "FFRec"
> conda install pytorch cudatoolkit=11.1 -c pytorch -c conda-forge 
> conda install -c conda-forge scikit-learn
> conda install -c conda-forge tqdm 
> conda install pandas
> conda install setprotitle
> pip install matplotlib
```

Hardware in our experiments:

* GPU: GeForce RTX 2080
* Cuda 11

## 1. Training

Move the experiment folder to your own target path, and modify the target path in corresponding files:
* ROOT path in 'data/preprocess.py'
* ROOT path in all 'XXX.sh' scripts.

Run for one of the four tasks {MF, FedMF, FairMF, FedFairMF}:
```
> bash run_XXX.sh
```

All models will be saved in 'models/' and training logs in 'logs/' under your target path.

## 2. Results

All results will be saved in 'results/' under your target path.

Observations and additional evaluations in the following notebooks:
* Training Observation: example of adding noise in F2MF, bounds of sigma, observing the fairness during as training curves
* Recommendation Performance: extract recommendation performances from log files to csv file, plot results over lambda
* Fairness Performance: evaluate group-wise performances and save to csv file, plot result over lambda
