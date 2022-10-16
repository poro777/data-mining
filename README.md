# data-mining
# Deep Kernel Learning for Clustering
[Paper](https://epubs.siam.org/doi/10.1137/1.9781611976236.72)
## Run
Run `main.ipynb` or `main.py`

## Modify
### Hyperparamter
- `main.ipynb` => hyperparamter
- `main.py` => `congif.py`

### Dataset
- `main.ipynb` => dataset
- `main.py` => `myDataSet`

### Network
- `main.ipynb` => network
- `main.py` => `network.py`

### Kernel
- `hsic.py`

### Tips
1. `N`: 通常越大越好，但要train更久，改`N`後`epoch`通常要調整。
2. `init_epoch`: 不用太大，f_function有點像就好，太像結果不會好。
3. `y_dim`: 可以是任意大於0的數字，結果未知。
4. `BATCH_SIZE`: 改`BATCH_SIZE`後learning rate通常要調整。
5. `LAMBDA`: 當embedding space的點跟原圖一樣，可以嘗試改小`LAMBDA`。
6. Loss: 通常total_loss要大，hsic_loss要大(0.01-0.05)，norm_loss要小。
7. totoal loop = `EPOCH` * `ITER`，可以任意調整。
