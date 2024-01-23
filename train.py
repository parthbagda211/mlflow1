
import tensorflow as tf
from sklearn.datasets import fetch_california_housing


import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature



from export_module import ExportModule
from l_regression import LinearRegression
from loss_function import mse_loss
from norm_fun import Normalizer
from data_prepare import data_pre 

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)



df= fetch_california_housing(as_frame=True)["frame"]

epochs = float(sys.argv[1]) if len(sys.argv) > 1 else 150
lr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001

if __name__ == "__main__":
    # Set a random seed for reproducible results
    tf.random.set_seed(42)

    # Load dataset
    
    x_train,x_test,y_train,y_test = data_pre(df)
    # Data normalization
    norm_x = Normalizer(x_train)
    norm_y = Normalizer(y_train)
    x_train_norm, y_train_norm = norm_x.norm(x_train), norm_y.norm(y_train)
    x_test_norm, y_test_norm = norm_x.norm(x_test), norm_y.norm(y_test)

    with mlflow.start_run():
        # Initialize linear regression model
        warnings.simplefilter('ignore')
        lin_reg = LinearRegression()

        # Use mini batches for memory efficiency and faster convergence
        batch_size = 32
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train_norm, y_train_norm))
        train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0]).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test_norm, y_test_norm))
        test_dataset = test_dataset.shuffle(buffer_size=x_test.shape[0]).batch(batch_size)

        # Set training parameters
        train_losses, test_losses = [], []

        # Format training loop
        for epoch in range(epochs):
            batch_losses_train, batch_losses_test = [], []

            # Iterate through the training data
            for x_batch, y_batch in train_dataset:
                with tf.GradientTape() as tape:
                    y_pred_batch = lin_reg(x_batch)
                    batch_loss = mse_loss(y_pred_batch, y_batch)
                # Update parameters with respect to the gradient calculations
                grads = tape.gradient(batch_loss, lin_reg.variables)
                for g, v in zip(grads, lin_reg.variables):
                    v.assign_sub(lr * g)
                # Keep track of batch-level training performance
                batch_losses_train.append(batch_loss)

            # Iterate through the testing data
            for x_batch, y_batch in test_dataset:
                y_pred_batch = lin_reg(x_batch)
                batch_loss = mse_loss(y_pred_batch, y_batch)
                # Keep track of batch-level testing performance
                batch_losses_test.append(batch_loss)

            # Keep track of epoch-level model performance
            train_loss = tf.reduce_mean(batch_losses_train)
            test_loss = tf.reduce_mean(batch_losses_test)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            if epoch % 10 == 0:
                mlflow.log_metric(key="train_losses", value=train_loss, step=epoch)
                mlflow.log_metric(key="test_losses", value=test_loss, step=epoch)
                print(f"Mean squared error for step {epoch}: {train_loss.numpy():0.3f}")

        # Log the parameters
        mlflow.log_params(
            {
                "epochs": epochs,
                "learning_rate": lr,
                "batch_size": batch_size,
            }
        )
        # Log the final metrics
        mlflow.log_metrics(
            {
                "final_train_loss": train_loss.numpy(),
                "final_test_loss": test_loss.numpy(),
            }
        )
        print(f"\nFinal train loss: {train_loss:0.3f}")
        print(f"Final test loss: {test_loss:0.3f}")

        # Export the tensorflow model
        lin_reg_export = ExportModule(model=lin_reg, norm_x=norm_x, norm_y=norm_y)

        # Infer model signature
        predictions = lin_reg_export(x_test)
        signature = infer_signature(x_test.numpy(), predictions.numpy())

        mlflow.tensorflow.log_model(lin_reg_export, "model", signature=signature)