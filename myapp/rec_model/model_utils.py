
from model_build import predict_movies

similarity, results_df = predict_movies('movie_embed', 8, 10, 100000)

print(results_df.title[:10].tolist())

    