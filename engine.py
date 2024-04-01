import chess
import torch
from train import Net, convert

# initialize engine and model
depth = 3
kmov = [[None, None] for _ in range(depth + 1)]
transposition = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# find best move at root of tree
def root(depth, game, maximizing):
  moves = order(game, depth)
  best = -9999
  found = None
  for move in moves:
    game.push(move)
    value = minimax(depth - 1, game, -10000, 10000, not maximizing)
    game.pop()
    if value >= best:
      best = value
      found = move
  return found

# minimax with alpha-beta pruning
def minimax(depth, game, alpha, beta, maximizing):
  fen = game.fen()
  if fen in transposition:
    se, sd = transposition[fen]
    if sd >= depth:
      return se
  if depth == 0:
    return -evaluate(game)
  moves = order(game, depth)
  if depth >= 2 and not maximizing:
    score = nmp(depth - 1, game, alpha, beta, maximizing)
    if score >= beta:
      return beta
  if maximizing:
    best = -9999
    for move in moves:
      game.push(move)
      best = max(best, minimax(depth - 1, game, alpha, beta, not maximizing))
      game.pop()
      alpha = max(alpha, best)
      if beta <= alpha:
        break
    return best
  else:
    best = 9999
    for move in moves:
      game.push(move)
      best = min(best, minimax(depth - 1, game, alpha, beta, not maximizing))
      game.pop()
      if best < beta:
        beta = best
        if not game.is_capture(move):
          km(move, depth)
      if beta <= alpha:
        break
    transposition[fen] = (best, depth)
    return best

# move ordering
def order(game, depth):
  killers = []
  promotions = []
  captures = []
  others = []
  if depth < len(kmov):
    for killer in kmov[depth]:
      if killer and game.is_legal(killer) and killer not in captures and killer not in promotions:
        killers.append(killer)
  for move in game.legal_moves:
    if move in killers:
      continue
    if game.is_capture(move):
      captures.append(move)
    elif move.promotion:
      promotions.append(move)
    else:
      others.append(move)
  return killers + promotions + captures + others

# killer moves
def km(move, depth):
  if move != kmov[depth][0]:
    kmov[depth][1] = kmov[depth][0]
    kmov[depth][0] = move

# null move pruning
def nmp(depth, game, alpha, beta, maximizing):
  null = chess.Move.null()
  game.push(null)
  score = -minimax(depth - 1, game, -beta, -alpha, not maximizing)
  game.pop()
  return score

# evaluate positions using neural network
def evaluate(game):
  bt = convert(game)
  bt = torch.tensor(bt, dtype=torch.float32).to(device)
  bt = bt.unsqueeze(0)
  with torch.no_grad():
    evaluation = model(bt).item()
  if game.turn == chess.BLACK:
    evaluation = -evaluation
  return evaluation
