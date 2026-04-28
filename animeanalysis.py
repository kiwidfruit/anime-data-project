MILESTONE 1:

import pandas as pd
import matplotlib.pyplot as plt

anime = pd.read_csv("anime.csv")

# REMOVE MISSING VARIABLES YES YES:)
anime = anime.dropna(subset=["rating"])

# FIX EPISODES COLUMN BECAUSE DATA WAS TOO BIGGGGGGG (*o*)
anime["episodes"] = pd.to_numeric(anime["episodes"], errors="coerce")
anime = anime.dropna(subset=["episodes"])

# wanted to remove extreme long anime so plot looks normal ;/
anime_filtered = anime[anime["episodes"] < 200]

# RATING DISTRIBUTION (0o0)
plt.figure()
plt.hist(anime["rating"], bins=30)
plt.title("Distribution of Anime Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")


# EPISODES DISTRIBUTION (U_U)
plt.figure()
plt.hist(anime_filtered["episodes"], bins=30)
plt.title("Distribution of Number of Episodes")
plt.xlabel("Episodes")
plt.ylabel("Frequency")

# ANIME TYPE COUNT (^v^)
plt.figure()
anime["type"].value_counts().plot(kind="bar")
plt.title("Count of Anime Types")
plt.xlabel("Type")
plt.ylabel("Count")


# MEMBERS (POPULARITY) <3
plt.figure()
plt.hist(anime["members"], bins=30)
plt.title("Distribution of Anime Popularity")
plt.xlabel("Members")
plt.ylabel("Frequency")


# MULTIVARIABLES (-_-)
plt.figure()
plt.scatter(anime["members"], anime["rating"])
plt.title("Rating vs Popularity")
plt.xlabel("Members")
plt.ylabel("Rating")

plt.figure()
plt.scatter(anime_filtered["episodes"], anime_filtered["rating"])
plt.title("Episodes vs Rating")
plt.xlabel("Episodes")
plt.ylabel("Rating")

MILESTONE 2:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

anime = pd.read_csv("anime.csv")

anime = anime.dropna(subset=["rating", "episodes", "members"])

anime = anime[anime["episodes"] != "Unknown"]
anime["episodes"] = anime["episodes"].astype(int)


anime = pd.get_dummies(anime, columns=["type"], drop_first=True)

X = anime.drop(["rating", "name", "genre", "anime_id"], axis=1)
y = anime["rating"]

anime.head()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model1 = LinearRegression()
model1.fit(X_train, y_train)

y_train_pred = model1.predict(X_train)
y_test_pred = model1.predict(X_test)

print("Linear Regression:")
print("R2:", r2_score(y_train, y_train_pred))
print("R2:", r2_score(y_test, y_test_pred))

X2 = anime[["members", "episodes"]]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2, y, test_size=0.2, random_state=42
)

model1b = LinearRegression()
model1b.fit(X_train2, y_train2)

y_test_pred2 = model1b.predict(X_test2)

r2_score(y_test2, y_test_pred2)

# versión 1
tree1 = DecisionTreeRegressor(max_depth=3)
tree1.fit(X_train, y_train)

y_pred_tree1 = tree1.predict(X_test)
r2_score(y_test, y_pred_tree1)

# versión 2
tree2 = DecisionTreeRegressor(max_depth=10)
tree2.fit(X_train, y_train)

y_pred_tree2 = tree2.predict(X_test)
r2_score(y_test, y_pred_tree2)


plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Linear Regression: Actual vs Predicted")

scores = [
    r2_score(y_test, y_pred_tree1),
    r2_score(y_test, y_pred_tree2)
]

labels = ["Depth 3", "Depth 10"]

plt.bar(labels, scores)
plt.title("Decision Tree R2 Comparison")
plt.ylabel("R2 Score")
