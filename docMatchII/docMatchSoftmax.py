import numpy as np
import sys
import tensorflow as tf

def main(_):
    # Import data
    trainX = np.loadtxt("data/docMatchIITrainX.txt")
    trainY = tf.one_hot(np.loadtxt("data/docMatchIITrainY.txt"), 3)
    devX = np.loadtxt("data/docMatchIIDevX.txt")
    devY = tf.one_hot(np.loadtxt("data/docMatchIIDevY.txt"), 3)
    testX = np.loadtxt("data/docMatchIITestX.txt")
    testY = tf.one_hot(np.loadtxt("data/docMatchIITestY.txt"), 3)

    # Create the model.
    x = tf.placeholder(tf.float32, [None, 2])
    W = tf.Variable(tf.zeros([2, 3]))
    b = tf.Variable(tf.zeros([3]))
    y = tf.matmul(x, W) + b

    # Output.
    y_ = tf.placeholder(tf.float32, [None, 3])

    # Loss.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    trainStep = tf.train.GradientDescentOptimizer(10).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    trainY = trainY.eval()
    devY = devY.eval()
    testY = testY.eval()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train
    print("Started Training")
    sys.stdout.flush()
    for i in range(70000):
        sess.run(trainStep, feed_dict={x: trainX, y_: trainY})
        if i % 1000 == 0:
            print("Train accuracy {}: {}".format(i, sess.run(accuracy, feed_dict={x: trainX, y_: trainY})))
            sys.stdout.flush()
    print("Finished Training")
    sys.stdout.flush()

    # Get dev error.
    print("Dev accuracy: {}".format(sess.run(accuracy, feed_dict={x: devX, y_: devY})))
    sys.stdout.flush()

    # Confusion matrix for dev.
    print(tf.contrib.metrics.confusion_matrix(tf.argmax(devY, 1), sess.run(tf.argmax(y, 1), feed_dict={x: devX})).eval())

    # Get test error.
    print("Test accuracy: {}".format(sess.run(accuracy, feed_dict={x: testX, y_: testY})))
    sys.stdout.flush()

if __name__ == '__main__':
    tf.app.run(main=main)
