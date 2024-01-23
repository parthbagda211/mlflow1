import tensorflow as tf

class Normalizer(tf.Module):

    def __init__(self,x):
        self.mean = tf.math.reduce_mean(x,axis=0)
        self.std = tf.math.reduce_std(x,axis=0)

    def norm(self,x):
        return (x-self.mean)/self.std 
    
    def unnorm(self,x):
        return (x*self.std)+self.mean 


