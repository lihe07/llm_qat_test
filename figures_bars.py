import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

sns.set_style(style="whitegrid")
sns.set_context("talk")

# 1. Parse the data provided by the user
csv_data = """Policy,Exact_After,F1_After,Exact_Drop,F1_Drop
Full Precision,0.4270,0.5015,0.2129,0.2334
Outlier Adaptive,0.4060,0.4896,0.1625,0.1857
Aggresive LoRA,0.4040,0.4828,0.1408,0.1707
Depth Adaptive,0.3740,0.4588,0.1576,0.1846
Conservative LoRA,0.3820,0.4600,0.1258,0.1662
Uniform LoRA (8-bit),0.4110,0.4847,0.1789,0.2013
Uniform LoRA (6-bit),0.2930,0.3685,0.0556,0.1034
Uniform LoRA (4-bit),0.0010,0.0413,-0.0001,0.0149
"""

df = pd.read_csv(io.StringIO(csv_data))

# 2. Calculate the 'Before Attack' performance scores
df["Exact_Before"] = df["Exact_After"] + df["Exact_Drop"]
df["F1_Before"] = df["F1_After"] + df["F1_Drop"]

# 3. Reshape the data into a "long-form" DataFrame suitable for Seaborn
before_df = df[["Policy", "Exact_Before", "F1_Before"]].copy()
before_df.rename(columns={"Exact_Before": "Exact", "F1_Before": "F1"}, inplace=True)
before_df["Stage"] = "Before"
before_df_long = before_df.melt(
    id_vars=["Policy", "Stage"], var_name="Metric", value_name="Score"
)

after_df = df[["Policy", "Exact_After", "F1_After"]].copy()
after_df.rename(columns={"Exact_After": "Exact", "F1_After": "F1"}, inplace=True)
after_df["Stage"] = "After"
after_df_long = after_df.melt(
    id_vars=["Policy", "Stage"], var_name="Metric", value_name="Score"
)

plot_df = pd.concat([before_df_long, after_df_long])


# 4. Create the plot using sns.catplot for faceting
g = sns.catplot(
    data=plot_df,
    x="Policy",
    y="Score",
    hue="Stage",
    col="Metric",  # This creates the separate plots for 'Exact' and 'F1'
    kind="bar",
    aspect=1.5,
    legend=False,
)

# 5. Customize the plots for clarity
for ax in g.axes.flat:
    ax.tick_params(axis="x", labelrotation=45)

g.axes.flat[0].set_title("Exact Score", fontsize=15)
g.axes.flat[1].set_title("F1 Score", fontsize=15)

plt.tight_layout()

# 6. Save the figure to a file
plt.savefig("./figures/policy_performance_drop.png")
