import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn import datasets as clustering
from py_code.utils import *
from py_code.config import *

class myDataSet(Dataset):
    def __init__(self,N, _type:str = 'moon'):
        if _type == 'moon':
            self.points, self.true_labels = clustering.make_moons(
                n_samples=N, noise=0.05, random_state=42
            )
            self.label_num = 2
        elif _type == 'circle':
            self.points, self.true_labels = clustering.make_circles(
                n_samples=N, factor=0.5, noise=0.05, random_state=42
            )
            self.label_num = 2
        elif _type == 'blob':
            self.points, self.true_labels = clustering.make_blobs(
                n_samples=N, cluster_std=[1.0, 2.5, 0.5], random_state=42
            )
            self.label_num = 3
        elif _type == 'aniso':
            self.points, self.true_labels = clustering.make_blobs(
                n_samples=N, cluster_std=[1.0, 2.5, 1], random_state=42
            )
            transformation = [[0.6, -0.6], [-0.3, 0.6]]
            self.points = np.dot(self.points, transformation)
            self.label_num = 3

        self.points = torch.DoubleTensor(self.points).to(device)
        self.labelColor = toColor(self.true_labels)
        self._type = _type
        
    def groupIndex(self,label):
        ''' given label return points index'''
        return self.true_labels == label

    def group(self, label):
        ''' given label return points'''
        return self.points[self.groupIndex(label)]

    def sample(self, N):
        ''' random sample N '''
        index = np.random.choice(self.__len__(), N, replace=False)
        return self.points[index], self.true_labels[index]

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]

# dataset
train_data = myDataSet(N, 'blob')
plot2D(to_numpy(train_data.points), train_data.labelColor,title='Points')
plt.savefig(IMAGE_PATH  +'original_points')
plt.clf()
dataSet = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True, drop_last=True)
