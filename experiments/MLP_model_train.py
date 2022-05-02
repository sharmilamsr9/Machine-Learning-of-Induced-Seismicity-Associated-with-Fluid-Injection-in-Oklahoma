import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from torchinfo import summary
from mlflow import log_metric, log_param, log_artifact
import mlflow
import textwrap

# Settings to change before the run
p = {
    "PARAM": "mu",  # mu or lambda
    "EXPERIMENT": "stage_1",
    "HIDDEN_SIZE": 64,  # 8 16 32 64
    "NUM_LAYERS": 2,  # 2 3 4 5
    "ACTIVATION": "sigmoid",  # relu, sigmoid
    "DROPOUT": 0.7,  # between 0 to 1
    "NEIGHBOR": True,  # True or False
    "HISTORIC": True,  # True or False
    "MONTHS": 12,  # 12, 6, 3
}

# code
experiment_name = "%s_%s" % (p["PARAM"], p["EXPERIMENT"])
experiment = mlflow.get_experiment_by_name(experiment_name)
if not experiment:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id
print("experiment_id", experiment_id)
mlflow.start_run(experiment_id=experiment_id)


# all settings in string
settings_string = ""
for k, v in p.items():
    log_param(k, v)
    settings_string += "%s_%s_" % (k, v)
# wrap long text
settings_string = "\n".join(textwrap.wrap(settings_string, 66))

df_train = pd.read_pickle("../old_pickles/my_df_train_%s.pickle" % (p["PARAM"]))
df_test = pd.read_pickle("../old_pickles/my_df_test_%s.pickle" % (p["PARAM"]))


for x in df_train.groupby(["lat", "lon"]):
    print(x[1])
    break


for x in df_test.groupby(["lat", "lon"]):
    print(x[1])
    break


x = {x[0] for x in df_train.groupby(["lat", "lon"])}
len(x)


x = {x[0] for x in df_test.groupby(["lat", "lon"])}
len(x)


def get_x_y(df):
    # train
    inj_vol = np.stack(df["inj_vol"].values)
    pp = np.stack(df["pp"].values)
    if not p["NEIGHBOR"]:
        # only consider original co-ordinate data and ignore neighbors
        inj_vol = inj_vol[:, :, 0]
        pp = pp[:, :, 0]

    inj_vol = inj_vol.reshape(inj_vol.shape[0], -1)
    pp = pp.reshape(pp.shape[0], -1)
    lat = np.stack(df["lat"].values)
    lat = np.reshape(lat, (-1, 1))
    lon = np.stack(df["lon"].values)
    lon = np.reshape(lon, (-1, 1))
    prev_data = np.stack(df["prev_" + p["PARAM"]].values)
    # (2814, 1) (2814, 1) (2814, 12, 1) (2814, 12, 1)
    # (2513, 1) (2513, 1) (2513, 12, 5) (2513, 12, 5)
    # dim: 2814 x 24
    if p["HISTORIC"]:
        x = np.concatenate([lat, lon, inj_vol, pp, prev_data], axis=1)
    else:
        x = np.concatenate([lat, lon, inj_vol, pp], axis=1)
    y = df[p["PARAM"]].values

    return x, y


# train
x_train, y_train = get_x_y(df_train)

# test
x_test, y_test = get_x_y(df_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

log_param("x_train.shape", x_train.shape)
log_param("y_train.shape", y_train.shape)
log_param("x_test.shape", x_test.shape)
log_param("y_test.shape", y_test.shape)


import torch
from torch import nn

torch.manual_seed(12345)

input_size = x_train.shape[1]
print("input_size = ", input_size)
log_param("input_size", input_size)

activations = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}


def build_mlp_layer(input_size, hidden_size, activation_layer, dropout_percentage):
    return [
        nn.Linear(input_size, hidden_size),
        activations[activation_layer],
        nn.Dropout(p=dropout_percentage),
    ]


layers = []
layers.extend(
    build_mlp_layer(input_size, p["HIDDEN_SIZE"], p["ACTIVATION"], p["DROPOUT"])
)
for _ in range(p["NUM_LAYERS"] - 1):
    layers.extend(
        build_mlp_layer(
            p["HIDDEN_SIZE"],
            p["HIDDEN_SIZE"],
            p["ACTIVATION"],
            p["DROPOUT"],
        )
    )
layers.append(nn.Linear(p["HIDDEN_SIZE"], 1))

model = nn.Sequential(*layers)

# convert train test data to torch tensors
X_train = torch.from_numpy(x_train.astype("float32"))
Y_train = torch.from_numpy(np.expand_dims(y_train, axis=-1).astype("float32"))
X_test = torch.from_numpy(x_test.astype("float32"))
Y_test = torch.from_numpy(np.expand_dims(y_test, axis=-1).astype("float32"))


print(X_train.shape, Y_train.shape, X_train.dtype, Y_train.dtype)
print(X_test.shape, Y_test.shape, X_test.dtype, Y_test.dtype)

lr = 0.001
log_param("Learning rate", lr)

# set random seed for initializing model weights
# model = Feedforward(x_train.shape[1], 32)
loss_fn = nn.MSELoss()  # +  weight regularization terms
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)


print(x_train.shape)
model_summary = summary(model, input_size=(1, input_size))
with open("model_summary.txt", "w") as f:
    f.write(str(model_summary))

log_artifact("model_summary.txt")


model.train()
loss_train_array = []
loss_test_array = []
epoch = 10000  # default: 10000
log_param("epoch", epoch)

for epoch in range(epoch):
    optimizer.zero_grad()
    # Forward pass
    loss_train = loss_fn(model(X_train), Y_train)
    loss_train_val = float(loss_train.detach().numpy())
    loss_train_array.append(loss_train_val)
    loss_test = loss_fn(model(X_test), Y_test)
    loss_test_val = float(loss_test.detach().numpy())
    loss_test_array.append(loss_test_val)

    log_metric("test_loss", loss_test_val)
    log_metric("train_loss", loss_train_val)

    print(
        "Epoch %s: train loss: %6.4f, test loss %6.4f"
        % (epoch, loss_train.item(), loss_test.item())
    )
    # Backward pass
    loss_train.backward()
    optimizer.step()


plt.figure(figsize=(8, 3), dpi=120)
plt.plot(loss_train_array)
plt.plot(loss_test_array)
plt.legend(["train loss", "test loss"])
plt.xlabel("epoch")
plt.ylabel("mse loss")
plt.title(settings_string)
# plt.ylim([0, 1])
# plt.show()
plt.savefig("train_test_loss.png", bbox_inches="tight")
log_artifact("train_test_loss.png")


loss_train_array[-1], loss_test_array[-1]


# # Training and testing for each cordinate


model.eval()
df_test["Pred"] = model(X_test).detach().numpy()
df_train["Pred"] = model(X_train).detach().numpy()

# save model
torch.save(model, "model.pt")
log_artifact("model.pt")
# train df
df_train.to_pickle("df_train_pred.pkl")
log_artifact("df_train_pred.pkl")
# test df
df_test.to_pickle("df_test_pred.pkl")
log_artifact("df_test_pred.pkl")

log_artifact(__file__)

df_train.head()


loss_co_ord = OrderedDict()
for co_ord, df_c in df_test.groupby(["lat", "lon"]):
    loss = (df_c["Pred"] - df_c[p["PARAM"]]) ** 2
    loss_co_ord[str(co_ord)] = loss
    # loss_co_ord.append(loss.detach().item())


plt.figure(figsize=(10, 3), dpi=120)
bar = plt.bar(loss_co_ord.keys(), [x.mean() for x in loss_co_ord.values()])
# plt.ylim([0, 1])
plt.xticks(rotation=90)
plt.legend([bar], ["Model loss \n per co-ordinate"])
# plt.title("Model loss per co-ordinate\n%s" % (settings_string))
plt.title(settings_string)
# plt.show()
plt.savefig("loss_per_coordinate.png", bbox_inches="tight")
log_artifact("loss_per_coordinate.png")


fig, axes = plt.subplots(4, 10, sharex=False, sharey=False, figsize=(14, 6), dpi=200)
axes = axes.ravel()


def plot_pred(axes, df_pred, column, color):
    max_data = 1
    df_pred = df_pred.groupby(["lat", "lon"])

    ax_idx = 0
    init = False
    for lon in np.arange(37.0, 36.2, -0.2):
        for lat in np.arange(99.2, 97.2, -0.2):
            ax = axes[ax_idx]
            lat_s = "%3.1f" % (lat)
            lon_s = "%3.1f" % (lon)
            # get df for given lat lon
            try:
                df = df_pred.get_group((round(lat, 1), round(lon, 1)))
                ax.plot(df.year_month, df[column], color)
                ax.text(
                    2011.00,
                    max_data - 0.2,
                    "%s\n%s" % (lat_s, lon_s),
                    fontsize=8,
                    color="r",
                )
            except KeyError as e:
                pass
                # print('Key error %s'%(e))

            # set injection volume limit
            ax.set_ylim([0, max_data])
            yticks = np.linspace(0, max_data, 3).astype("int")
            ax.set_yticks(yticks)
            ax.set_xticks([2011.00, 2018.92])

            ax.set_yticklabels([])
            ax.set_xticklabels([])
            # injection volume y axis label
            if lat_s == "99.2" and lon_s == "36.6":
                ax.set_yticklabels(yticks, color="k")
                ax.set_ylabel(p["PARAM"], color="k")

            if lat_s == "97.6" and lon_s == "36.4":
                ax.set_xticklabels([2011, 2018])

            ax_idx += 1


plot_pred(axes, df_train, p["PARAM"], "limegreen")
plot_pred(axes, df_train, "Pred", "slategrey")

plot_pred(axes, df_test, p["PARAM"], "limegreen")
plot_pred(axes, df_test, "Pred", "darkorange")


# fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
fig.suptitle(settings_string)
# plt.show()
plt.savefig("Overlayed_predictions.png")
log_artifact("Overlayed_predictions.png")

mlflow.end_run()
