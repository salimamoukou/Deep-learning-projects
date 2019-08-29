import tensorflow as tf
import numpy as np


class Network:
    """
    Neural net class -- set params and can launch the training.
    Many other improvment are comming (loading, inference, testing, validation, saving function - Lossploting function - definie global var
    with parser....etc)
    """
    def __init__(self, archi):
        self._archi = archi
        self._batchSize = 32
        self._epoch = 1
        self._optimizer = None
        self._learningRate = 0.01
        self._featuresDims = 8

    def set_inputs_placeholders(self):
        self._inputs_holder = tf.placeholder(tf.float32, shape=(None, 50, self._featuresDims))
        self._labels_holder = tf.placeholder(tf.float32, shape=(None, 50))

    def set_arch(self):
        self._archi.buildNetwork(self)

    def set_output(self):
        self._outputs = self._archi._outputs

    def cost_function(self, lab, out):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=lab, logits=out)

    def set_loss(self):
        self._loss = self.cost_function(self._labels, self._outputs)

    def set_data_iterator(self):
        data = tf.data.Dataset.from_tensor_slices(self._inputs_holder)
        label = tf.data.Dataset.from_tensor_slices(self._labels_holder)
        train_dataset = tf.data.Dataset.zip((data,label))
        train_dataset.batch(self._batchSize)
        self._dataIterator = train_dataset.make_initializable_iterator()
        self._inputs, self._labels = self._dataIterator.get_next()

    def set_data_shape(self):
        self._numSamples = tf.shape(self._inputs_holder)[0]

    def set_training(self):
        optimizer = tf.train.AdamOptimizer(self._learningRate)
        self.trainOP = optimizer.minimize(self._loss)

    def set_save(self):
        self._saver = tf.train.Saver()

    def train(self, inp, out):
        with tf.Graph().as_default():

            self.set_inputs_placeholders()
            self.set_data_iterator()

            self.set_arch()
            self.set_output()

            self.set_loss()
            self.set_training()
            self.set_save()

            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run([init])

            for i in range(self._epoch):
                sess.run(self._dataIterator.initializer, feed_dict={self._inputs_holder: inp,
                                                                    self._labels_holder: out})

                for j in range(int(np.ceil(out.shape[0]/self._batchSize))):
                    sess.run([self._inputs,self._labels])
                    print(sess.run(self._loss))
                    sess.run(self.trainOP)
