import numpy as np
import torch
from torch.nn import functional as F

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

from py_code.utils import *
from py_code.hsic import *
from py_code.network import *
from py_code.dataset import *

def f_function(x):
    global encoder, decoder
    return decoder(encoder(x))

def test_network():
    global encoder, train_data
    x = train_data.sample(3)
    with torch.no_grad():
        print(x)
        print(encoder(x))
        print(f_function(x))

def init_weight(INIT_EPOCH = 10, ENCODER_LR = 0.01, DECODER_LR = 0.005):
    global encoder, decoder, dataSet

    encoderOptimizer = torch.optim.Adam(encoder.parameters(),lr=ENCODER_LR)
    decoderOptimizer = torch.optim.Adam(decoder.parameters(),lr=DECODER_LR)

    record_loss = []
    # train mode
    encoder.train()
    decoder.train()
    tqdm_ = tqdm(range(INIT_EPOCH))
    for i in tqdm_:
        for points in dataSet:
            encoderOptimizer.zero_grad()
            decoderOptimizer.zero_grad()
            
            loss = norm(points - encoder(points)) + \
                norm(points - f_function(points))
            loss.backward()
            
            encoderOptimizer.step()
            decoderOptimizer.step()
        tqdm_.set_postfix({'loss':loss.item()})

        record_loss.append(loss.item())

    plt.plot(record_loss)

    plt.savefig(IMAGE_PATH + 'init_loss')
    plt.clf()
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(3)
    fig.tight_layout()
    plot2D(to_numpy(encoder(train_data.points)), train_data.labelColor, title='embedding', axes=121)
    plot2D(to_numpy(f_function(train_data.points)), train_data.labelColor, title='f_function', axes=122)
    plt.savefig(IMAGE_PATH + 'init_graph')
    plt.close(fig)

def check_U_mode():
    sample, sampleLabel = train_data.sample(BATCH_SIZE)
    U = to_numpy(update_U(sample))
    print('suggest y_dim = batch_size in check_U')
    while True:
        dim = [int(i) for i in input('dim: ').split(' ')]
        color = toColor(sampleLabel)
        if len(dim) == 1: # 1d
            x = U[:,dim[0]]
            y = sampleLabel
            plt.scatter(x, y, c = color)

        elif len(dim) == 2: # 2d
            x = U[:,dim[0]]
            y = U[:,dim[1]]
            plt.scatter(x, y, c = color)

        else: # 3d
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            x = U[:, dim[0]]
            y = U[:, dim[1]]
            z = U[:, dim[2]]
            ax.scatter( x, y, z, c = color)

        plt.show()

def update_U(points, debug = False):
    global hsic, y_dim
    embedding_k = hsic._kernel_x(encoder(points))
    H = hsic.H
    D = np.diag(to_cpu(embedding_k).mean(1))
    D = torch.tensor(fractional_matrix_power(np.linalg.inv(D), 0.5).real,device=device)
    L =  H @ D @ embedding_k @ D @ H
    # eigenvalue
    _, eigenvectors = np.linalg.eig(to_cpu(L))
    # top C
    U = eigenvectors[:,:y_dim].real

    if debug: # check (U^T) * U = I
        print(U.transpose() @ U)

    return torch.DoubleTensor(U).to(device)

def centerLoss(data, center):
    labels = []
    for x in to_numpy(data):
        labels.append(np.argmin(abs(center - x).sum(axis=1)))

    cluster = torch.tensor(labels)
    cluster = F.one_hot(cluster.long(),num_classes=len(center) ).double().to(device)
    #mean = x.T @ cluster / cluster.sum(dim=0)
    #x_cluster_mean = (mean @ cluster.T).T
    #x - x_cluster_mean
    return norm(data - ((torch.nan_to_num((data.detach().T @ cluster) / cluster.sum(dim=0)) @ cluster.T).T))

def train(EPOCH = 20, ITER = 5, ENCODER_LR = 0.01, DECODER_LR = 0.05, SCHEDULER_STEP = 5, SCHEDULAR_GAMMA = 0.99):
    global encoder, decoder, dataSet, hsic

    encoderOptimizer = torch.optim.Adam(encoder.parameters(),lr=ENCODER_LR)
    decoderOptimizer = torch.optim.Adam(decoder.parameters(),lr=DECODER_LR)

    encode_LR_Scheduler = torch.optim.lr_scheduler.StepLR(encoderOptimizer, SCHEDULER_STEP, SCHEDULAR_GAMMA)
    decoder_LR_Scheduler = torch.optim.lr_scheduler.StepLR(decoderOptimizer, SCHEDULER_STEP, SCHEDULAR_GAMMA)


    hsic_loss = []
    norm_loss = []
    record_loss = []
    center_loss = []
    image_set = []
    # train mode
    encoder.train()
    decoder.train()
    tqdm_ = tqdm(range(EPOCH))
    for i in tqdm_:
        kmeans = KMeans(n_clusters= K_CENTER, n_init=3,random_state=0, max_iter=30).fit(to_numpy(encoder(train_data.points)))
        k_center = kmeans.cluster_centers_
        for points in dataSet:
            # update D, U
            hsic.update_D(hsic._kernel_x(points))
            U = update_U(points)
            # D U fixed, update ~theta
            for _ in range(ITER):
                encoderOptimizer.zero_grad()
                decoderOptimizer.zero_grad()

                x = encoder(points)
                _hsic = hsic(x, U)
                _norm = LAMBDA * norm(points - f_function(points))

                centerloss_ = CENTER * centerLoss(x, k_center) if WITH_CENTER_LOSS else torch.zeros(1,device=device)
                loss = _hsic - _norm - centerloss_
                # SGA
                loss = -loss
                loss.backward()
                
                encoderOptimizer.step()
                decoderOptimizer.step()

        decoder_LR_Scheduler.step()
        encode_LR_Scheduler.step()

        CENTER.step()
        LAMBDA.step()

        center_loss.append(centerloss_.item())
        hsic_loss.append(_hsic.item())
        norm_loss.append(_norm.item())
        record_loss.append(-(loss.item()))
        tqdm_.set_postfix({'loss': -(loss.item())})

        if SAVE_GIF and i % SAVE_STEP == 0:
            image_set.append(plot_result(U = update_U(points),title= str(i)))
    if SAVE_GIF:
        image_set[0].save(IMAGE_PATH  + 'result.gif',save_all=True, append_images=image_set[1:], optimize=False, duration=200, loop=0)

    # plot
    fig = plt.figure()
    fig.set_figwidth(16)
    fig.set_figheight(3)
    fig.tight_layout()
    plt.subplot(141)
    plt.title("loss")
    plt.plot(record_loss)
    plt.subplot(142)
    plt.ticklabel_format(useOffset=False)
    plt.title("hsic loss")
    plt.plot(hsic_loss)
    plt.subplot(143)
    plt.title("norm loss")
    plt.plot(norm_loss)
    plt.subplot(144)
    plt.title("center loss")
    plt.plot(center_loss)
    plt.savefig(IMAGE_PATH + 'training_loss')
    plt.close(fig)

def plot_result(U = None, title = ''):
    global encoder, train_data, hsic
    
    fig = plt.figure()
    fig.set_figwidth(16)
    fig.set_figheight(10)
    fig.tight_layout()

    # sorted index
    index = np.argsort(train_data.true_labels)

    points = train_data.points[index]
    colors = train_data.labelColor[index]
    with torch.no_grad():
        embedding_points = encoder(points)

    # original points
    plot2D(to_numpy(points), colors, title='true_label', axes=231)
    # U
    if U is not None: plot2D(to_numpy(U), title='U(sample)', axes=233)
    # embedding points
    plot2D(to_numpy(embedding_points), colors, title='embedding', axes=234)
    # decode(encode(points))
    plot2D(to_numpy(f_function(train_data.points)), train_data.labelColor, title='f_function', axes=236)
    # original kernel
    kx = hsic._kernel_x(points)
    plotKernelMatrix(to_numpy(kx), colors, title='Kx', axes=232)
    # embedding kernel
    embedding_k = hsic._kernel_x(embedding_points)
    plotKernelMatrix(to_numpy(embedding_k), colors, title='Kψ(x,θ)', axes=235)

    plt.savefig(IMAGE_PATH + 'training_result')
    fig.suptitle(title, fontsize=16)
    img = None
    if SAVE_GIF:    
        img = fig2img(fig)

    plt.close(fig)

    return img

def plot_distribution(axis = 0):
    global train_data
    
    fig, axes = plt.subplots(2, train_data.label_num, figsize=(4*train_data.label_num, 8))
    for i in range(train_data.label_num):
        axes[0,i].set_title(f'original label {i} axis-{axis}')
        axes[1,i].set_title(f'embedding label {i} axis-{axis}')

        sns.histplot(to_numpy(train_data.points[train_data.groupIndex(i),axis]),ax=axes[0,i])
        sns.histplot(to_numpy(encoder(train_data.points))[train_data.groupIndex(i),axis],ax=axes[1,i],bins=20)
    plt.savefig(IMAGE_PATH + 'dist')
    plt.close(fig)

def apply_kmeans(k, data, title='title'):
    global train_data
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    print(title,end=': ')
    acc = normalized_mutual_info_score(train_data.true_labels,kmeans.labels_)
    print(acc)

    plot2D(to_numpy(train_data.points),color=toColor(kmeans.labels_) , title=title)
    plt.savefig(IMAGE_PATH + f'k_means_{title}')
    plt.clf()
    
    return acc

def accuracy():
    global encoder, train_data
    apply_kmeans(train_data.label_num, to_numpy(train_data.points), 'oringal')
    acc = apply_kmeans(train_data.label_num, to_numpy(encoder(train_data.points)), 'embedding')
    return acc