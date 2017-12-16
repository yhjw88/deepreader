import numpy as np
import sys
import tensorflow as tf

def main(_):
    # Import data
    trainX = np.loadtxt("data/docMatchITrainX.txt")
    trainY = np.loadtxt("data/docMatchITrainY.txt")
    trainY = [0 if i == -1 else 1 for i in trainY]
    devX = np.loadtxt("data/docMatchIDevX.txt")
    devY = np.loadtxt("data/docMatchIDevY.txt")
    devY = [0 if i == -1 else 1 for i in devY]
    testX = np.loadtxt("data/docMatchITestX.txt")
    testY = np.loadtxt("data/docMatchITestY.txt")
    testY = [0 if i == -1 else 1 for i in testY]

    # Create the model.
    x = tf.placeholder(tf.float32, [None, 2])

    # Hidden Layer.
    with tf.variable_scope('layer1'):
        W1 = tf.get_variable('w1', [2, 3], initializer=tf.random_normal_initializer())
        b1 = tf.get_variable('b1', [1], initializer=tf.constant_initializer(0.0))
        y1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

    # Output Layer.
    with tf.variable_scope('layer2'):
        W2 = tf.get_variable('w2',[3, 1], initializer= tf.random_normal_initializer())
        b2 = tf.get_variable('b2',[1], initializer=tf.constant_initializer(0.0))
        y2 = tf.matmul(y1, W2) + b2

    # Output.
    y = tf.squeeze(y2)
    y_ = tf.placeholder(tf.float32, [None])

    # Loss.
    alpha = 0.0001
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
    l2 = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
    trainStep = tf.train.GradientDescentOptimizer(10).minimize(cross_entropy + alpha * l2)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # For testing.
    correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(y)), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train
    print("Started Training")
    sys.stdout.flush()
    for i in range(6000):
        sess.run(trainStep, feed_dict={x: trainX, y_: trainY})
        if i % 1000 == 0:
            print("Train accuracy: {}".format(sess.run(accuracy, feed_dict={x: trainX, y_: trainY})))
            sys.stdout.flush()
    print("Finished Training")
    sys.stdout.flush()

    # Test trained model.
    print("Dev accuracy: {}".format(sess.run(accuracy, feed_dict={x: devX, y_: devY})))
    sys.stdout.flush()

    # Test trained model again.
    print("Test accuracy: {}".format(sess.run(accuracy, feed_dict={x: testX, y_: testY})))
    sys.stdout.flush()

if __name__ == '__main__':
    tf.app.run(main=main)
