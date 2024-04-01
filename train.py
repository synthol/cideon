import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# neural network architecture
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(12, 16, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.fc1 = nn.Linear(64 * 2 * 2, 256)
    self.fc2 = nn.Linear(256, 1)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2)
    x = F.relu(self.conv3(x))
    x = x.view(-1, 64 * 2 * 2)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return torch.tanh(x)

# process games and generate training set
class Data(Dataset):
  def __init__(self, paths):
    self.positions = []
    self.results = []
    loaded = 0
    for path in paths:
      with open(path) as file:
        while True:
          game = chess.pgn.read_game(file)
          if game is None:
            break
          result = game.headers['Result']
          if result not in ['1-0', '0-1', '1/2-1/2']:
            continue
          board = game.board()
          for move in game.mainline_moves():
            board.push(move)
            self.positions.append(convert(board))
            self.results.append(self.numeric(result))
          loaded += 1
          if loaded % 1000 == 0:
            print(f'loaded: {loaded}')
  
  def numeric(self, result):
    if result == '1-0':
      return 1
    elif result == '0-1':
      return -1
    else:
      return 0

  def __len__(self):
    return len(self.positions)

  def __getitem__(self, idx):
    return self.positions[idx], self.results[idx]

# convert positions to numerical format
def convert(board):
  bt = np.zeros((12, 8, 8), dtype=np.float32)
  pm = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
  for square in chess.SQUARES:
    piece = board.piece_at(square)
    if piece is not None:
      row, col = divmod(square, 8)
      index = pm[piece.piece_type]
      bt[index + (6 if piece.color != chess.WHITE else 0), row, col] = 1
  return bt

# load dataset and prepare dataloader
def retrieve(paths, batch_size=64):
  dataset = Data(paths)
  return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# train model
def train(paths, epochs=3, batch_size=64, lr=0.001, pretrained=None):
  dataloader = retrieve(paths, batch_size=batch_size)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = Net().to(device)
  if pretrained:
    model.load_state_dict(torch.load(pretrained, map_location=device))
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  model.train()
  lowest = float('inf')
  for epoch in range(epochs):
    rl = 0.0
    for i, data in enumerate(dataloader, 0):
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs.squeeze(), labels.float())
      loss.backward()
      optimizer.step()
      rl += loss.item()
      if i % 100 == 99:
        average = rl / 100
        print(f'epoch: {epoch + 1}, it: {i + 1}, loss: {rl / 100:.3f}')
        if average < lowest:
          lowest = average
          save = 'model.pth' if pretrained is None else 'updated.pth'
          torch.save(model.state_dict(), save)
          print(f'saved: {lowest:.3f}')
        rl = 0.0

if __name__ == '__main__':
  paths = ['dataset.pgn']
  train(paths)
  # uncomment to train using pretrained model
  # train(paths, pretrained='model.pth')
