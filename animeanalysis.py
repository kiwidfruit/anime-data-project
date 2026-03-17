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
plt.show()

# EPISODES DISTRIBUTION (U_U)
plt.figure()
plt.hist(anime_filtered["episodes"], bins=30)
plt.title("Distribution of Number of Episodes")
plt.xlabel("Episodes")
plt.ylabel("Frequency")
plt.show()

# ANIME TYPE COUNT (^v^)
plt.figure()
anime["type"].value_counts().plot(kind="bar")
plt.title("Count of Anime Types")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()

# MEMBERS (POPULARITY) <3
plt.figure()
plt.hist(anime["members"], bins=30)
plt.title("Distribution of Anime Popularity")
plt.xlabel("Members")
plt.ylabel("Frequency")
plt.show()

# MULTIVARIABLES (-_-)
plt.figure()
plt.scatter(anime["members"], anime["rating"])
plt.title("Rating vs Popularity")
plt.xlabel("Members")
plt.ylabel("Rating")
plt.show()

plt.figure()
plt.scatter(anime_filtered["episodes"], anime_filtered["rating"])
plt.title("Episodes vs Rating")
plt.xlabel("Episodes")
plt.ylabel("Rating")
