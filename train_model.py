import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load generated training data
df = pd.read_csv("ipl_2025_generated_data.csv")

# Official IPL 10 teams
teams = [
    'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Delhi Capitals', 'Kolkata Knight Riders', 'Sunrisers Hyderabad',
    'Rajasthan Royals', 'Punjab Kings', 'Lucknow Super Giants',
    'Gujarat Titans'
]

# Official IPL 10 venues
venues = [
    'Wankhede Stadium', 'Eden Gardens', 'M. Chinnaswamy Stadium',
    'Arun Jaitley Stadium', 'MA Chidambaram Stadium', 'Narendra Modi Stadium',
    'Rajiv Gandhi Intl. Stadium', 'Sawai Mansingh Stadium',
    'Punjab Cricket Association Stadium', 'Ekana Cricket Stadium'
]

# ✅ 1. Map old team names to current official teams
df["batting_team"] = df["batting_team"].replace({
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Rising Pune Supergiant": "Chennai Super Kings",
    "Rising Pune Supergiants": "Chennai Super Kings",
})

df["bowling_team"] = df["bowling_team"].replace({
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Rising Pune Supergiant": "Chennai Super Kings",
    "Rising Pune Supergiants": "Chennai Super Kings",
})

# ✅ 2. Drop rows with teams/venues not in your final lists
df = df[df["batting_team"].isin(teams) & df["bowling_team"].isin(teams)]
df = df[df["venue"].isin(venues)]

print("Unique batting teams:", df["batting_team"].unique())
print("Unique bowling teams:", df["bowling_team"].unique())
print("Unique venues:", df["venue"].unique())

# ✅ 3. Drop rows with missing venue
df = df.dropna(subset=["venue"])

# ✅ 4. Label encode using full official lists
team_encoder = LabelEncoder()
venue_encoder = LabelEncoder()

team_encoder.fit(teams)
venue_encoder.fit(venues)

df["batting_team"] = team_encoder.transform(df["batting_team"])
df["bowling_team"] = team_encoder.transform(df["bowling_team"])
df["venue"] = venue_encoder.transform(df["venue"])

# ✅ 5. Build features (make sure runs_last_5_overs exists!)
X = df[[
    "batting_team", "bowling_team", "venue", "over_ball",
    "current_score", "wickets",
    "runs_left", "balls_left", "crr", "rrr"
]]

y = df["result"]

print("✅ X shape:", X.shape)
print("✅ y unique:", y.unique())

# ✅ 6. Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ 7. Save model & encoders
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("team_encoder.pkl", "wb") as f:
    pickle.dump(team_encoder, f)
with open("venue_encoder.pkl", "wb") as f:
    pickle.dump(venue_encoder, f)

print("✅ Model + encoders saved for 10 teams + 10 venues!")





