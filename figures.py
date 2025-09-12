from matplotlib.patches import bbox_artist
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

# Use seaborn style
sns.set_theme(style="whitegrid")
sns.set_context("talk")

# Data
policies = [
    "FullPrecision",
    "OutlierAdaptive",
    "Aggressive",
    "DepthAdaptive",
    "ConservativeLoRA",
    "Uniform(8)",
    "Uniform(6)",
    "Uniform(4)",
]
bitbudgets = [
    2720612416,
    463372864,
    423838528,
    545576512,
    590168128,
    682180672,
    512311360,
    342442048,
]


def bits_to_MB(bits):
    return bits / 8 / (1024 * 1024)


mb_budgets = np.array([bits_to_MB(b) for b in bitbudgets])
exact_scores = np.array(
    [
        0.5943235572374646,
        0.5684957426679281,
        0.5448438978240303,
        0.5315988647114475,
        0.5077578051087985,
        0.5808893093661306,
        0.3486281929990539,
        0.0008514664143803217,
    ]
)
f1_scores = np.array(
    [
        0.7002484148756903,
        0.6753506433518305,
        0.6535098448899275,
        0.643391701663484,
        0.6262254536882265,
        0.6867220171120404,
        0.4718602772535547,
        0.05615857881741222,
    ]
)


def pareto_front(x, y):
    # Lower x is better, higher y is better
    points = np.array(sorted(zip(x, y, range(len(x))), key=lambda t: t[0]))
    pareto = []
    max_y = -np.inf
    for xi, yi, idx in points:
        if yi > max_y:
            pareto.append((xi, yi, idx))
            max_y = yi
    return np.array(pareto)


def non_pareto_front(x, y):
    # Lower x is better, higher y is better
    points = np.array(sorted(zip(x, y, range(len(x))), key=lambda t: t[0]))
    non_pareto = []
    max_y = -np.inf
    for xi, yi, idx in points:
        if yi <= max_y:
            non_pareto.append((xi, yi, idx))
        else:
            max_y = yi
    return np.array(non_pareto)


pareto_exact = pareto_front(mb_budgets, exact_scores)
pareto_f1 = pareto_front(mb_budgets, f1_scores)

non_pareto_exact = non_pareto_front(mb_budgets, exact_scores)
non_pareto_f1 = non_pareto_front(mb_budgets, f1_scores)

# Marker map for policies (unique shapes)
policy_markers = {
    "FullPrecision": "o",
    "OutlierAdaptive": "s",
    "Aggressive": "^",
    "DepthAdaptive": "D",
    "ConservativeLoRA": "P",
    "Uniform(8)": "v",
    "Uniform(6)": "X",
    "Uniform(4)": "*",
}

fig, ax = plt.subplots(figsize=(10, 6))

# Plot all points faintly
sc_all_exact = ax.scatter(
    non_pareto_exact[:, 0],
    non_pareto_exact[:, 1],
    color="C0",
    alpha=0.3,
    label="All Policies (Exact)",
)
sc_all_f1 = ax.scatter(
    non_pareto_f1[:, 0],
    non_pareto_f1[:, 1],
    color="C1",
    alpha=0.3,
    label="All Policies (F1)",
)

# Plot Pareto front lines (no markers here; we add shaped markers separately)
(ln_pf_exact,) = ax.plot(
    pareto_exact[:, 0],
    pareto_exact[:, 1],
    color="C0",
    lw=1.5,
    label="Pareto Front (Exact)",
)
(ln_pf_f1,) = ax.plot(
    pareto_f1[:, 0],
    pareto_f1[:, 1],
    color="C1",
    lw=1.5,
    label="Pareto Front (F1)",
)

# policy-shaped markers
front_policy_indices_exact = [int(idx) for _, _, idx in pareto_exact]

front_policy_indices_f1 = [int(idx) for _, _, idx in pareto_f1]

marker_edge = "black"
marker_size = 80

for idx in front_policy_indices_exact:
    pol = policies[idx]
    ax.scatter(
        mb_budgets[idx],
        exact_scores[idx],
        color="C0",
        marker=policy_markers[pol],
        s=marker_size,
        edgecolor=marker_edge,
        linewidth=0.5,
        zorder=3,
    )

for idx in front_policy_indices_f1:
    pol = policies[idx]
    ax.scatter(
        mb_budgets[idx],
        f1_scores[idx],
        color="C1",
        marker=policy_markers[pol],
        s=marker_size,
        edgecolor=marker_edge,
        linewidth=0.5,
        zorder=3,
    )

# Build a legend for the policy marker shapes (only those that appear on any Pareto front)
policies_on_front = []
for idx in sorted(set(front_policy_indices_exact + front_policy_indices_f1)):
    policies_on_front.append(policies[idx])

policy_handles = [
    Line2D(
        [0],
        [0],
        marker=policy_markers[pol],
        color="w",
        label=pol,
        markerfacecolor="lightgray",
        markeredgecolor=marker_edge,
        markersize=8,
        linestyle="",
    )
    for pol in policies_on_front
]

# First legend: datasets/lines
legend1 = ax.legend(
    handles=[sc_all_exact, sc_all_f1],
    loc="lower right",
    frameon=True,
    title="Series",
    bbox_to_anchor=(1, 0.25),
)
ax.add_artist(legend1)

# Second legend: policy marker shapes
ax.legend(
    handles=policy_handles,
    loc="lower right",
    ncol=min(3, len(policy_handles)),
    frameon=True,
    title="Pareto Front Policies",
    # bbox_to_anchor=(0.5, 1),
)

ax.set_xlabel("Bit Budget (MB)")
ax.set_ylabel("Score")
ax.grid(True)
fig.tight_layout()

# save
plt.savefig("figures/pareto_fronts.png", dpi=300)
