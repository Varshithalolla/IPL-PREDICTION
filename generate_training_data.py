import pandas as pd
import numpy as np

# Load CSVs
deliveries = pd.read_csv("deliveries.csv")
matches = pd.read_csv("matches.csv")

print("✅ Deliveries columns:", deliveries.columns.tolist())
print("✅ Matches columns:", matches.columns.tolist())

print("🔢 Example match_id from deliveries:", deliveries["match_id"].unique()[:5])
print("🔢 Example id from matches:", matches["id"].unique()[:5])

# Merge to get venue
merged = deliveries.merge(
    matches[["id", "venue"]],
    left_on="match_id",
    right_on="id",
    how="left"
)

print("✅ After merge, venues:", merged["venue"].unique())
print("✅ Merged shape before features:", merged.shape)

# Drop rows with no venue
merged = merged.dropna(subset=["venue"])
print("✅ Merged shape after cleaning:", merged.shape)
print("✅ Venues left:", merged["venue"].unique())

# Add features (ONLY ONCE)
merged["current_score"] = merged.groupby("match_id")["total_runs"].cumsum()
merged["target_runs"] = 200
merged["runs_left"] = merged["target_runs"] - merged["current_score"]

merged["over_ball"] = merged["over"] + merged["ball"] / 6
merged["over_ball"] = merged["over_ball"].replace(0, 0.1)  # Prevent div by zero

merged["balls_left"] = 120 - (merged["over"] * 6 + merged["ball"])
merged["balls_left"] = merged["balls_left"].replace(0, 1)  # Prevent div by zero

merged["wickets"] = merged.groupby("match_id")["is_wicket"].cumsum()
merged["crr"] = merged["current_score"] / merged["over_ball"]
merged["rrr"] = merged["runs_left"] / (merged["balls_left"] / 6)
merged["result"] = merged["inning"].apply(lambda x: 1 if x == 2 else 0)

# Replace inf/-inf with NaN, then drop those rows
merged = merged.replace([np.inf, -np.inf], np.nan)
merged = merged.dropna(subset=["crr", "rrr"])

# Save
merged.to_csv("ipl_2025_generated_data.csv", index=False)
print("✅ Saved: ipl_2025_generated_data.csv")

