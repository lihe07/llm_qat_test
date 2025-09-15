import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    "Policy": [
        "Full",
        "Outlier Adaptive",
        "Aggresive",
        "Depth adaptive",
        "Conservative",
        "Uniform 8",
        "UniformLoRAPolicy(6,6,32,True,6)",
        "UniformLoRAPolicy(4,4,32,True,4)",
    ],
    "Exact": [0.4270, 0.4060, 0.4040, 0.3740, 0.3820, 0.4110, 0.2930, 0.0010],
    "F1": [0.5015, 0.4896, 0.4828, 0.4588, 0.4600, 0.4847, 0.3685, 0.0413],
    "Exact Drop": [0.2129, 0.1625, 0.1408, 0.1576, 0.1258, 0.1789, 0.0556, -0.0001],
    "F1 Drop": [0.2334, 0.1857, 0.1707, 0.1846, 0.1662, 0.2013, 0.1034, 0.0149],
}
df = pd.DataFrame(data)

# 2. Reshape the data for Seaborn
# We "melt" the DataFrame to make it "long-form", which is ideal for creating grouped bar plots.
df_melted = df.melt(
    id_vars=["Policy"],
    value_vars=["Exact Drop", "F1 Drop"],
    var_name="Metric",
    value_name="Drop Value",
)

# 3. Create the plot
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 8))

# Order the policies by the 'F1 Drop' for a more informative visualization
order = df.sort_values("F1 Drop", ascending=False)["Policy"]

# Draw the bar plot
sns.barplot(
    data=df_melted, x="Policy", y="Drop Value", hue="Metric", order=order, ax=ax
)

# 4. Customize the plot for clarity
ax.set_title("Performance Drop Across Different Policies", fontsize=16, weight="bold")
ax.set_xlabel("Policy", fontsize=12)
ax.set_ylabel("Performance Drop (Higher is Worse)", fontsize=12)
ax.legend(title="Metric")

# Rotate x-axis labels to prevent them from overlapping
plt.xticks(rotation=45, ha="right")

# Ensure all elements fit into the figure area
plt.tight_layout()

# 5. Save the figure to a file
plt.savefig("./figures/policy_performance_drop.png")
