import numpy
import pandas
import matplotlib.pyplot as plt
# Generate a data set with spirals
# http://cs231n.github.io/neural-networks-case-study/

def generate_spirals(N = 1000):
    K = 3 # number of classes
    N = N // K # number of points per class
    D = 2 # dimensionality
    
    data = numpy.zeros((N*K,D)) # data matrix (each row = single example)
    labels = numpy.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
      ix = range(N*j,N*(j+1))
      r = numpy.linspace(0.0,2,N) # radius
      a = 4.15
      t = numpy.linspace(j*a +1,(j+1)*a-0.5,N) + numpy.random.randn(N)*0.2 # theta
      data[ix] = numpy.c_[r*numpy.sin(t), r*numpy.cos(t)]
      labels[ix] = j
    
    # save outer poinet
    outer = abs(data) >= 0.2
    outer = numpy.logical_or(outer[:,0], outer[:,1])
    return data[outer], labels[outer]

def save_spirals(data, labels):
    # Save to a csv file
    f = open('spirals.csv', 'w')
    f.write('x,y,label\n')
    for i in range(len(labels)):
        f.write(str(data[i][0]) + ',' + str(data[i][1]) + ',' + str(labels[i]) + '\n')
    f.close()

# Visualize data set
def visualize_data_set():
    
    # Load data set
    ds = pandas.read_csv('spirals.csv')
    # Print first 5 rows in data set
    print('--- First 5 rows ---')
    print(ds.head())
    # Print the shape
    print('\n--- Shape of data set ---')
    print(ds.shape)
    # Print class distribution
    print('\n--- Class distribution ---')
    print(ds.groupby('label').size())
    # Visualize data set
    figure = plt.figure(figsize = (12, 8))
    figure.suptitle('Spirals', fontsize=16)
    grouped_dataset = ds.groupby('label')
    labels = ['0', '1', '2']
    for i, group in grouped_dataset:
        plt.scatter(group['x'], group['y'], label=labels[int(i)])
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend()
    #plt.show()
    plt.savefig('spirals.png')
# Generate spirals

if __name__ == '__main__':
    save_spirals(*generate_spirals())
    # Visualize data set
    visualize_data_set()