import tensorflow as tf 

class ExportModule(tf.Module):

    def __init__(self,model,norm_x,norm_y):
        
        self.model = model
        self.norm_x = norm_x
        self.norm_y = norm_y 

    @tf.function(input_signature=[tf.TensorSpec(shape=[None,None],dtype=tf.float32)])

    def __call__(self,x):

        x = self.norm_x.norm(x)
        y = self.model(x)
        y = self.norm_y.unnorm(y)
        return y 