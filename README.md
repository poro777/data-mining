# data-mining final
# Deep Kernel Learning for Clustering
[Paper](https://epubs.siam.org/doi/10.1137/1.9781611976236.72)

# Requirements 
Python
- pytorch
- numpy
- tqdm
- seaborn
- sklearn
- matplotlib
- pandas

# Run
## Run
```
python main.py
```

## Hyperparameter
`py_code/config.py`

### Dataset
| Dataset type | Source |
| --- | --- |
| `moon` | sklearn.datasets |
| `circle` | sklearn.datasets |
| `blob` | sklearn.datasets |
| `aniso` | sklearn.datasets |
| `spiral` | py_code/spiral.py ([from](http://cs231n.github.io/neural-networks-case-study/))

## Output 
### file
- log.txt
- encoder model
### diagram
- Training loss
![](./demo/training_loss.png)

- Training result (embedding points)
![](./demo/training_result.png)

- Clustering results using k-means
![](./demo/k_means_embedding.png)