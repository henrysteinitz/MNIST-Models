from cajal.flow import *
import gzip, pickle
import numpy as np

# Load data
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()

# build model
vector_size = len(train_set[0][0])
one_hot_labels = [np.zeros(10) for _ in train_set[1]]
for i, label in enumerate(train_set[1]):
    one_hot_labels[i][label] = 1

mlp = Flow(inputs='x', outputs='y')
mlp.connect_variable(name='x', sinks=['product1'])
mlp.connect_parameter(name='W1', sinks=['product1'],
                      shape=(11, vector_size))
mlp.connect_map(name='product1', map=matrix_vector_product,
                sources=['W1', 'x'], sink='p1')
mlp.connect_variable(name='p1', source='product1', sinks=['sum1'])
mlp.connect_parameter(name='b1', sinks=['sum1'],
                      shape=(11,))
mlp.connect_map(name='sum1', map=add, sources=['p1', 'b1'], sink='s1')
mlp.connect_variable(name='s1', source='sum1', sinks=['sigmoid'])
mlp.connect_map(name='sigmoid', map=sigmoid, sources=['s1'], sink='h')
mlp.connect_variable(name='h', source='sigmoid', sinks=['product2'])
mlp.connect_parameter(name='W2', sinks=['product2'],
                      shape=(10, 11))
mlp.connect_map(name='product2', map=matrix_vector_product, sources=['W2', 'h'], sink='p2')
mlp.connect_variable(name='p2', source='product2', sinks=['sum2'])
mlp.connect_parameter(name='b2', sinks=['sum2'], shape=(10,))
mlp.connect_map(name='sum2', map=add, sources=['p2', 'b2'], sink='s2')
mlp.connect_variable(name='s2', source='sum2', sinks=['sigmoid2'])
mlp.connect_map(name='sigmoid2', map=soft_max, sources=['s2'], sink='y')
mlp.connect_variable(name='y', source='sigmoid2')
mlp.set_loss(sources=['y'], scalar_map=l2_norm, supervisors=1)
mlp.initialize_parameters()

# train and test
for k in range(200):
    count = 0
    mlp.train(inputs={'x': train_set[0][k*250:((k+1)*250)]},
              outputs=one_hot_labels[k*250:((k+1)*250)],
              learning_rate=.15)
    for i in range(len(test_set[0])):
        result = mlp.play(input_values={'x': test_set[0][i]})
        if np.argmax(result['y']) == test_set[1][i]:
            count += 1
    print('{}: {}'.format(k, count / len(test_set[0])))

count = 0
for k in range(len(test_set[0])):
    result = mlp.play(input_values={'x': test_set[0][k]})
    if np.argmax(result['y']) == test_set[1][k]:
        count += 1
print(count / len(test_set[0]))
