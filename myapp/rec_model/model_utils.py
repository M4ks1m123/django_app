from model_build import predict_books

similarity, results_df = predict_books('book_embed', 8, 10, 100000)

print(results_df.title[:10].tolist())