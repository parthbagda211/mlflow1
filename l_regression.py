import tensorflow as tf 

class LinearRegression(tf.Module):

    def __init__(self):
        self.built = False 
    
    @tf.function 

    def __call__(self,x):

        if not self.built:
            
            rand_w = tf.random.uniform(shape=[x.shape[-1],1])
            rand_b = tf.random.uniform(shape=[])
            self.w = tf.Variable(rand_w)
            self.b = tf.Variable(rand_b)
            self.built = True 
        y = tf.add(tf.matmul(x,self.w),self.b)

        return tf.squeeze(y,axis=1)