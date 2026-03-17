import pandas as pd
import matplotlib.pyplot as plt

anime = pd.read_csv("anime.csv")

# REMOVE MISSING VARIABLES YES YES:)
anime = anime  .dropna(subset=["rating"])

# RATING DISTRIBUTION (0o0)
plt.hist(anime["rating"], bins=30)
plt.title("Distribution of Anime Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")


# EPISODES DISTRIBUTION (U_U)
plt.hist(anime["episodes"], bins=30)
plt.title("Distribution of Number of Episodes")
plt.xlabel("Episodes")
plt.ylabel("Frequency")


# ANIME TYPE COUNT (^v^)
anime["type"].value_counts().plot(kind="bar")
plt.title("Count of Anime Types")
plt.xlabel("Type")
plt.ylabel("Count")


# MEMBERS (POPULARITY) <3
plt.hist(anime["members"], bins=30)
plt.title("Distribution of Anime Popularity")
plt.xlabel("Members")
plt.ylabel("Frequency")


# MULTIVARIABLES (-_-)
plt.scatter(anime["members"], anime["rating"])
plt.title("Rating vs Popularity")
plt.xlabel("Members")
plt.ylabel("Rating")

plt.scatter(anime["episodes"], anime["rating"])
plt.title("Episodes vs Rating")
plt.xlabel("Episodes")
plt.ylabel("Rating")
