import torch
from torch import nn


class RecommendationModel(nn.Module):
    
    def __init__(self, n_users, n_books,
                 n_factors=50, embedding_dropout=0.02,
                 hidden=10, dropouts=0.2):
        super().__init__()
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_books, n_factors)
        self.drop = nn.Dropout(embedding_dropout)
        self.hidden = nn.Sequential(nn.Linear(n_factors * 2, 128),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    # nn.Linear(2048, 512),
                                    # nn.ReLU(),
                                    # nn.Dropout(0.2),
                                    # nn.Linear(512, 128),
                                    # nn.ReLU(),
                                    # nn.Dropout(0.2)
                                    )
        self.fc = nn.Linear(128, 1)

    def forward(self, users, books, minmax=None):
        features = torch.cat([self.u(users), self.m(books)], dim=1)
        x = self.drop(features)
        x = self.hidden(x)
        out = torch.sigmoid(self.fc(x))
        if minmax is not None:
            min_rating, max_rating = minmax
            out = out*(max_rating - min_rating + 1) + min_rating - 0.5
        return out