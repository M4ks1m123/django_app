
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import math
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt


from data_processing import batches, create_dataset
from load_data import load_data

from model_config import RecommendationModel

ratings, books = load_data()

def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)
        
RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)


(n, m), (X, y), _ = create_dataset(ratings)
# print(f'Embeddings: {n} users, {m} books')
# print(f'Dataset shape: {X.shape}')
# print(f'Target shape: {y.shape}')

recommendation_model = RecommendationModel(
    n_users=n, n_books=m,
    n_factors=20, hidden=[500],
    embedding_dropout=0.05, dropouts=[0.25])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
datasets = {'train': (X_train, y_train), 'val': (X_valid, y_valid)}
dataset_sizes = {'train': len(X_train), 'val': len(X_valid)}

lr = 1e-4
wd = 1e-5
bs = 2000
n_epochs = 100
patience = 10
no_improvements = 0
best_loss = np.inf
best_weights = None
history = []
lr_history = []

minmax = ratings.rating.min(), ratings.rating.max()
minmax

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

recommendation_model.to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(recommendation_model.parameters(), lr=lr, weight_decay=wd)
iterations_per_epoch = int(math.ceil(dataset_sizes['train'] // bs))
# scheduler = CyclicLR(optimizer, cosine(t_max=iterations_per_epoch * 2, eta_min=lr/10))

for epoch in range(n_epochs):
    stats = {'epoch': epoch + 1, 'total': n_epochs}

    for phase in ('train', 'val'):
        training = phase == 'train'
        running_loss = 0.0
        n_batches = 0
        batch_num = 0
        for batch in batches(*datasets[phase], shuffle=training, bs=bs):
            x_batch, y_batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
            # compute gradients only during 'train' phase
            with torch.set_grad_enabled(training):
                outputs = recommendation_model(x_batch[:, 0], x_batch[:, 1], minmax)
                loss = criterion(outputs, y_batch)

                # don't update weights and rates when in 'val' phase
                if training:
                    # scheduler.step()
                    loss.backward()
                    optimizer.step()
                    # lr_history.extend(scheduler.get_lr())

            running_loss += loss.item()

        epoch_loss = running_loss / dataset_sizes[phase]
        stats[phase] = epoch_loss

        # early stopping: save weights of the best model so far
        if phase == 'val':
            if epoch_loss < best_loss:
                print('loss improvement on epoch: %d' % (epoch + 1))
                best_loss = epoch_loss
                best_weights = copy.deepcopy(recommendation_model.state_dict())
                no_improvements = 0
            else:
                no_improvements += 1

    history.append(stats)
    print('[{epoch:03d}/{total:03d}] train: {train:.4f} - val: {val:.4f}'.format(**stats))
    if no_improvements >= patience:
        print('early stopping after epoch {epoch:03d}'.format(**stats))
        break
    
ax = pd.DataFrame(history).drop(columns='total').plot(x='epoch')
ax.plot()
plt.show()

ground_truth, predictions = [], []

with torch.no_grad():
    for batch in batches(*datasets['val'], shuffle=False, bs=bs):
        x_batch, y_batch = [b.to(device) for b in batch]
        outputs = recommendation_model(x_batch[:, 0], x_batch[:, 1], minmax)
        # print(outputs)
        ground_truth.extend(y_batch.tolist())
        predictions.extend(outputs.tolist())

ground_truth = np.asarray(ground_truth).ravel()
predictions = np.asarray(predictions).ravel()

final_loss = np.sqrt(np.mean((np.array(predictions) - np.array(ground_truth))**2))
print(f'Final RMSE: {final_loss:.4f}')

MODEL_PATH = Path("C:/Users/Caster/Desktop/django_app/mysite/myapp/rec_model/model_state_dict/")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'recommendation_model.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f'Saving model to: {MODEL_SAVE_PATH}')
torch.save(obj=recommendation_model.state_dict(), f=MODEL_SAVE_PATH)