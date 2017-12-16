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

    # Hidden Layer.
    with tf.variable_scope('layer1'):
        W1 = tf.get_variable('w1', [2, 3], initializer=tf.random_normal_initializer())
        b1 = tf.get_variable('b1', [1], initializer=tf.constant_initializer(0.0))
        y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

    # Output Layer.
    with tf.variable_scope('layer2'):
        W2 = tf.get_variable('w2',[3, 3], initializer= tf.random_normal_initializer())
        b2 = tf.get_variable('b2',[3], initializer=tf.constant_initializer(0.0))
        y2 = tf.matmul(y1, W2) + b2

    # Output.
    y = y2
    y_ = tf.placeholder(tf.float32, [None, 3])

    # Loss.
    alpha = 0.0001
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    l2 = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
    trainStep = tf.train.GradientDescentOptimizer(1).minimize(cross_entropy + alpha * l2)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    trainY = trainY.eval()
    devY = devY.eval()

    # For testing.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train
    print("Started Training")
    sys.stdout.flush()
    for i in range(15000):
        sess.run(trainStep, feed_dict={x: trainX, y_: trainY})
        if i % 1000 == 0:
            print("Train accuracy {}: {}".format(i, sess.run(accuracy, feed_dict={x: trainX, y_: trainY})))
            sys.stdout.flush()
    print("Finished Training")
    sys.stdout.flush()

    # Get dev error.
    print("Dev accuracy: {}".format(sess.run(accuracy, feed_dict={x: devX, y_: devY})))
    sys.stdout.flush()


if __name__ == '__main__':
    tf.app.run(main=main)
