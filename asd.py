audio_cols = ["energy", "acousticness", "loudness", "instrumentalness", "danceability", "valence"]                                                                         
candidates = ["industrial", "metal", "heavy-metal", "death-metal", "metalcore", "rock", "alt-rock", "hard-rock"]
df.groupby("track_genre")[audio_cols].median().loc[candidates].round(3)