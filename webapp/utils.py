from cProfile import label
import matplotlib.pyplot as plt
import numpy as np


def plot_pred(ax, df, column):
    # Reduce 10% to min so we get to see the min values in plot
    mi = min(df["Pred"].min(), df[column].min()) * 0.9
    # Add 10% extra to max so we get to see the max values in plot
    # and legends don't overlap with data
    ma = max(df["Pred"].max(), df[column].max()) * 1.1

    df_train = df[df.test == False]
    df_test = df[df.test == True]

    ax.plot(df.year_month, df[column], label="Original", color="black")
    # train
    ax.plot(
        df_train.year_month,
        df_train["Pred"],
        label="Predicted-train",
        color="limegreen",
    )
    # test
    ax.plot(
        df_test.year_month,
        df_test["Pred"],
        label="Predicted-test",
        color="gold",
    )
    ax.legend()
    # set injection volume limit
    ax.set_ylim([mi, ma])
    yticks = np.linspace(mi, ma, 5).round(3)
    ax.set_yticks(yticks)
    ax.set_xticks(np.arange(2011, 2019, 1))
    # ax.set_xticks([2011.00, 2018.92])

    ax.set_yticklabels(yticks, color="k")
    ax.set_ylabel(column, color="k")

    ax.set_xticklabels(np.arange(2011, 2019, 1))


def plot_input(ax, df):
    # inj_vol is of dim 12 x 5, where zeroth column belongs to center lat lon, and
    # other 4 belong to neighbor lat lon. So we select zeroth column.
    # 12 months of data and the last value represents the current year_month, hence select -1 index
    inj_vol = df.inj_vol.apply(lambda x: x[-1, 0])
    pp = df.pp.apply(lambda x: x[-1, 0])

    ax.plot(df.year_month, inj_vol, label="Injection volume", color="k")

    # set injection volume limit
    # Reduce 10% to min, and increase 10% to max so we get to see the min, max values in plot
    mi, ma = inj_vol.min() * 0.9, inj_vol.max() * 1.1
    ax.set_ylim([mi, ma])
    yticks = np.linspace(mi, ma, 5).round(3)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, color="k")
    ax.set_ylabel("Injection volume (in barrel units)", color="k")

    ax.legend(loc="upper left")

    # twin the x axis to plot on the right side
    ax = ax.twinx()
    ax.scatter(
        df.year_month, pp, marker="^", label="Pore pressure", color="violet", s=30
    )

    mi, ma = pp.min() * 0.9, pp.max() * 1.1
    ax.set_ylim([mi, ma])
    yticks = np.linspace(mi, ma, 5).round(3)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, color="violet")
    ax.set_ylabel("Pore pressure (in MPa)", color="violet")

    ax.legend(loc="upper right")

    ax.set_xticks(np.arange(2011, 2019, 1))
    ax.set_xticklabels(np.arange(2011, 2019, 1))
