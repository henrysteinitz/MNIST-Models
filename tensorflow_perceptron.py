import tensorflow as tf
import gzip, pickle
import numpy as np

# Load data
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()
one_hot_labels = [np.zeros(10) for _ in train_set[1]]
for i, label in enumerate(train_set[1]):
    one_hot_labels[i][label] = 1

# build model
sess = tf.InteractiveSession()
vector_size = len(train_set[0][0])

x = tf.placeholder(dtype=tf.float32, shape=[None, vector_size])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([vector_size, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

sess.run(tf.global_variables_initializer())
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# train & test
for k in range(500):
    train_step.run(feed_dict={
        x: train_set[0][k*100:(k+1)*100],
        y_: one_hot_labels[k*100:(k+1)*100],
    })

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_set_labels = [np.zeros(10) for _ in test_set[1]]
for i, label in enumerate(test_set[1]):
    test_set_labels[i][label] = 1

print(accuracy.eval(feed_dict={
    x: test_set[0],
    y_: test_set_labels
}))
