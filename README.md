# FedFairRec
Federated Fairness-aware Recommendation

## 0. Setup

Code environment:

```
> conda create -n bmrl python=3.9
> conda activate bmrl
> conda install -c anaconda ipykernel
> python -m ipykernel install --user --name bmrl --display-name "BMRL"
> conda install pytorch cudatoolkit=11.1 -c pytorch -c conda-forge 
> conda install -c conda-forge scikit-learn
> conda install -c conda-forge tqdm 
> conda install pandas
> pip install setprotitle
```

Hardware:

* GPU: GeForce RTX 2080
* Cuda 11.2