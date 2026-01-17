import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Load raw training data to get PRICE column
dataset_path = os.path.join("datasets", "Resights_Hackathon_Ejerlejligheder_TRAIN.csv")

print("Loading training data...")
try:
    df = pd.read_csv(dataset_path, sep=",", encoding="utf-8")
except (pd.errors.ParserError, UnicodeDecodeError):
    df = pd.read_csv(dataset_path, sep=",", engine="python", encoding="utf-8")

# Extract prices and remove NaN values
prices = df["PRICE"].dropna().values
log_prices = np.log(prices)

print(f"Loaded {len(prices)} samples")
print(f"Price range: {prices.min():,.0f} - {prices.max():,.0f} DKK")
print(f"Mean price: {prices.mean():,.0f} DKK")
print(f"Median price: {np.median(prices):,.0f} DKK")

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Original price distribution (histogram)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(prices, bins=100, edgecolor="black", alpha=0.7, color="steelblue")
ax1.set_xlabel("Price (DKK)", fontsize=12)
ax1.set_ylabel("Frequency", fontsize=12)
ax1.set_title(
    "Original Price Distribution (Right-Skewed)", fontsize=14, fontweight="bold"
)
ax1.grid(True, alpha=0.3)
# Add statistics
ax1.axvline(
    prices.mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {prices.mean():,.0f}",
)
ax1.axvline(
    np.median(prices),
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Median: {np.median(prices):,.0f}",
)
ax1.legend()

# 2. Log-transformed price distribution (should be normal)
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(log_prices, bins=100, edgecolor="black", alpha=0.7, color="coral")
ax2.set_xlabel("Log(Price)", fontsize=12)
ax2.set_ylabel("Frequency", fontsize=12)
ax2.set_title(
    "Log-Transformed Price Distribution (Normal)", fontsize=14, fontweight="bold"
)
ax2.grid(True, alpha=0.3)
# Add normal distribution overlay
mu, sigma = log_prices.mean(), log_prices.std()
x_norm = np.linspace(log_prices.min(), log_prices.max(), 100)
y_norm = stats.norm.pdf(x_norm, mu, sigma) * len(log_prices) * (x_norm[1] - x_norm[0])
ax2.plot(x_norm, y_norm, "r-", linewidth=2, label=f"Normal(μ={mu:.2f}, σ={sigma:.2f})")
ax2.legend()

# 3. Q-Q plot for original prices (should show deviation from normal)
ax3 = fig.add_subplot(gs[1, 0])
stats.probplot(prices, dist="norm", plot=ax3)
ax3.set_title("Q-Q Plot: Original Prices vs Normal", fontsize=14, fontweight="bold")
ax3.grid(True, alpha=0.3)

# 4. Q-Q plot for log prices (should be close to normal)
ax4 = fig.add_subplot(gs[1, 1])
stats.probplot(log_prices, dist="norm", plot=ax4)
ax4.set_title("Q-Q Plot: Log Prices vs Normal", fontsize=14, fontweight="bold")
ax4.grid(True, alpha=0.3)

# 5. Density comparison: original vs log
ax5 = fig.add_subplot(gs[2, :])
# Original prices (normalized for comparison)
sns.kdeplot(data=prices, ax=ax5, label="Original Prices", linewidth=2, alpha=0.6)
# Log prices (exponentiated back for comparison)
exp_log_prices = np.exp(log_prices)
sns.kdeplot(
    data=exp_log_prices, ax=ax5, label="Exp(Log Prices)", linewidth=2, alpha=0.6
)
ax5.set_xlabel("Price (DKK)", fontsize=12)
ax5.set_ylabel("Density", fontsize=12)
ax5.set_title(
    "Density Comparison: Original vs Log-Normal Distribution",
    fontsize=14,
    fontweight="bold",
)
ax5.legend()
ax5.grid(True, alpha=0.3)

# Add overall title
fig.suptitle(
    "Demonstrating Log-Normal Distribution of Property Prices",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)

# Save figure
output_dir = Path("viz_output")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "price_distribution_log_normal.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\nVisualization saved to: {output_path}")

# Print statistics
print("\n" + "=" * 60)
print("STATISTICAL TESTS")
print("=" * 60)

# Test for normality of log prices
shapiro_stat, shapiro_p = stats.shapiro(log_prices[:5000])  # Limit for Shapiro-Wilk
print("\nShapiro-Wilk test for log prices (n=5000):")
print(f"  Statistic: {shapiro_stat:.4f}")
print(f"  p-value: {shapiro_p:.4f}")
print(
    f"  {'Log prices appear normal' if shapiro_p > 0.05 else 'Log prices may not be normal'}"
)

# Test for normality of original prices (should fail)
shapiro_stat_orig, shapiro_p_orig = stats.shapiro(prices[:5000])
print("\nShapiro-Wilk test for original prices (n=5000):")
print(f"  Statistic: {shapiro_stat_orig:.4f}")
print(f"  p-value: {shapiro_p_orig:.4f}")
print(
    f"  {'Original prices appear normal' if shapiro_p_orig > 0.05 else 'Original prices are NOT normal (expected)'}"
)

# Skewness and kurtosis
print("\nOriginal prices:")
print(f"  Skewness: {stats.skew(prices):.4f} (positive = right-skewed)")
print(f"  Kurtosis: {stats.kurtosis(prices):.4f}")

print("\nLog prices:")
print(f"  Skewness: {stats.skew(log_prices):.4f} (close to 0 = symmetric)")
print(f"  Kurtosis: {stats.kurtosis(log_prices):.4f} (close to 0 = normal)")

plt.show()
