import tensorflow as tf
from sklearn.datasets import fetch_california_housing


# df = fetch_california_housing(as_frame=True)['frame']
def data_pre(df):
    df = df.dropna()

    df = df[:1500]

    df_tf =tf.convert_to_tensor(df,dtype=tf.float32)

    #split data 

    df_shuffled = tf.random.shuffle(df_tf,seed=42)
    train_df,test_df = df_shuffled[100:],df_shuffled[:100]

    x_train,y_train = train_df[:,:-1],train_df[:,-1]
    x_test,y_test = test_df[:,:-1],test_df[:,-1]


    return x_train,x_test,y_train,y_test






