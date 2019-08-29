import tensorflow as tf


class NetworkArchi:
    """
    The global architecture (encoder and decoder) --- His principal goal is to set self._output
    """
    def __init__(self):
        self._encoder = None
        self._decoder = None

    def conv2d(self, input, units):
        return tf.layers.dense(inputs=input, units=units)
        # You need to watch the doc of this function (it does layers dense only on the last dimension...)

    def encoder(self, network):
        inputs = network._inputs
        enco = self.conv2d(inputs, 64)
        enco = self.conv2d(enco, 128)
        latents = self.conv2d(enco, 256)
        # output : BatchSize x Ligne x 256
        return latents

    def decoder(self, latents):
        deco = self.conv2d(latents, 256)
        deco = self.conv2d(deco, 128)
        deco = self.conv2d(deco, 64)
        deco = self.conv2d(deco, 1)
        deco = tf.reshape(deco, shape=[50])
        # output : BatchSize x Ligne x 50
        return deco

    def buildNetwork(self, network):
        latents = self.encoder(network)
        self._outputs = self.decoder(latents)
