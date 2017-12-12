import numpy as np
import sys
import tensorflow as tf

def main(_):
    # Import data
    trainX = np.loadtxt("data/docMatchIITrainX.txt")
    trainY = tf.one_hot(np.loadtxt("data/docMatchIITrainY.txt"), 3)
    devX = np.loadtxt("data/docMatchIIDevX.txt")
    devY = tf.one_hot(np.loadtxt("data/docMatchIIDevY.txt"), 3)

    # Create the model.
    x = tf.placeholder(tf.float32, [None, 2])
    W = tf.Variable(tf.zeros([2, 3]))
    b = tf.Variable(tf.zeros([3]))
    y = tf.matmul(x, W) + b

    # # layer 1
    # with tf.variable_scope('layer1'):
    #     W1 = tf.get_variable('w1', [2, 10], initializer=tf.random_normal_initializer())
    #     b1 = tf.get_variable('b1', [1], initializer=tf.constant_initializer(0.0))
    #     y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

    # # layer 2
    # with tf.variable_scope('layer2'):
    #     W2 = tf.get_variable('w2',[10, 1], initializer= tf.random_normal_initializer())
    #     b2 = tf.get_variable('b2',[1], initializer=tf.constant_initializer(0.0))
    #     y2 = tf.matmul(y1, W2) + b2

    # output
    y_ = tf.placeholder(tf.float32, [None, 3])

    # logisticLoss = tf.reduce_mean(tf.log(1 + tf.exp(-y_ * y)))
    # cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    trainStep = tf.train.GradientDescentOptimizer(10).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    trainY = trainY.eval()
    devY = devY.eval()

    # Train
    print("Started Training")
    sys.stdout.flush()
    for _ in range(80000):
      sess.run(trainStep, feed_dict={x: trainX, y_: trainY})
    print("Finished Training")
    sys.stdout.flush()

    # Get train error.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Train Error: {}".format(sess.run(accuracy, feed_dict={x: trainX, y_: trainY})))

    # Get dev error.
    print("Dev Error: {}".format(sess.run(accuracy, feed_dict={x: devX, y_: devY})))
    sys.stdout.flush()


if __name__ == '__main__':
    tf.app.run(main=main)
