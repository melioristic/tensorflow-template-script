import tensorflow as tf

class NeuralNetwork(tf.keras.Model):
  
  def __init__(self, **kwargs):
    super(NeuralNetwork, self).__init__(**kwargs)
    
    # Initialise the layers here
    self.l1 = #
    self.l2 = #
  
  def call(self, input_tensor):
    x = self.l1(input_tensor)
    x = self.l2(x)
    
    return x 
  
  def model(self, input_shape):
    x =  tf.keras.layers.Input(shape = input_shape)
    
    return tf.keras.Model(inputs = [x], outputs = self.call(x))
    
    
