import tensorflow as tf
import gzip, pickle
import numpy as np
from matplotlib import pyplot as plt


# Load data
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()
one_hot_labels = [np.zeros(10) for _ in train_set[1]]
for i, label in enumerate(train_set[1]):
    one_hot_labels[i][label] = 1

sess = tf.InteractiveSession()
vector_size = len(train_set[0][0])
hidden_layer_size = 300

x = tf.placeholder(dtype=tf.float32, shape=[None, vector_size])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])

W1 = tf.Variable(tf.random_normal([vector_size, hidden_layer_size], stddev=0.35))
b1 = tf.Variable(tf.random_normal([hidden_layer_size], stddev=0.35))

h = tf.sigmoid(tf.matmul(x, W1) + b1)

W2 = tf.Variable(tf.random_normal([hidden_layer_size, 10], stddev=0.35))
b2 = tf.Variable(tf.random_normal([10], stddev=0.35))

y = tf.matmul(h, W2) + b2

sess.run(tf.global_variables_initializer())
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.GradientDescentOptimizer(3.0).minimize(loss)

for k in range(500):
    print(k)
    train_step.run(feed_dict={
        x: train_set[0][k*100:(k+1)*100],
        y_: one_hot_labels[k*100:(k+1)*100],
    })

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_set_labels = [np.zeros(10) for _ in test_set[1]]
for i, label in enumerate(test_set[1]):
    test_set_labels[i][label] = 1

print(accuracy.eval(feed_dict={
    x: test_set[0],
    y_: test_set_labels
}))


# Daydream
noise = np.array([np.zeros(784)])
digit = 2
image_gradient = tf.gradients([tf.slice(y, [0, digit], [1, 1])], [x])
print(image_gradient)
for k in range(40000):
    if (k%100 == 0):
        print(k)
    step = image_gradient[0].eval(feed_dict={x: noise})
    noise += step*(100.0)

plt.imshow(noise.reshape([28,28]), interpolation='nearest')
plt.show()
