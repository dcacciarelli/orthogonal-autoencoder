import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from model import OrthogonalAutoEncoder
from training_utils import TrainingModel, create_grouped_bar_plot
from gradients import get_integrated_gradients, get_integrated_gradients_mc, get_integrated_gradients_rect

# AI4I (data from https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
# only keeping columns: Air_temperature, Process_temperature, Rotational_speed, Torque, Tool_wear, y
df = pd.read_csv("/Users/dcac/Data/ai4i/ai4i2020_reduced.csv").iloc[:1000, 3:]
df_train = df[df["y"] == 0].drop(["y"], axis=1)  # clean training data to fit the model
df_test = df[df["y"] == 1].drop(["y"], axis=1)

# Scaling data
scaler = StandardScaler()
x1 = torch.Tensor(pd.DataFrame(scaler.fit_transform(df_train)).values)
x2 = torch.Tensor(pd.DataFrame(scaler.transform(df_train)).values)


# Initiating model
model = OrthogonalAutoEncoder(encoding_layers=[5, 5], activation_function=nn.ReLU(), dropout=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(model)
sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training model (validation loss is lower because of dropout)
train = TrainingModel(penalty=0.1, learning_rate=0.001, batch_size=20, val_size=0.20, verbose=True)
train_loader, valid_loader = train.create_datasets(train_data=x1)
trained_model, losses = train.train_model(model, patience=10, n_epochs=1000)

# Plot losses
plt.figure(figsize=(10, 10))
plt.plot(losses["train"], label="training loss")
plt.plot(losses["val"], label="validation loss")
plt.legend()
plt.show()

# Saving trained model
# torch.save(trained_model, "orthogonal_autoencoder.pt")

# Computing integrated gradients of the encoder network
base = x1.mean(axis=0)
example = x2[23]
int_grads_tr, _ = get_integrated_gradients(model=trained_model, inp=example, baseline=base, steps=20)
int_grads_mc, _ = get_integrated_gradients_mc(model=trained_model, inp=example, baseline=base, samples=100)

# Plotting integrated gradients
create_grouped_bar_plot(df_train.columns, Trapezoidal=int_grads_tr, Monte_Carlo=int_grads_mc)
