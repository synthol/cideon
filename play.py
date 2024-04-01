import threading
import pygame
import chess
import engine
from pygame.locals import *

# initialize game
pygame.init()
width = 810
height = 640
color = (150, 110, 80)
square_size = 80
win = pygame.display.set_mode((width, height))
pygame.display.set_caption('cideon')
lock = threading.Lock()
wp = pygame.image.load('img/wp.png')
bp = pygame.image.load('img/bp.png')
wn = pygame.image.load('img/wn.png')
bn = pygame.image.load('img/bn.png')
wb = pygame.image.load('img/wb.png')
bb = pygame.image.load('img/bb.png')
wr = pygame.image.load('img/wr.png')
br = pygame.image.load('img/br.png')
wq = pygame.image.load('img/wq.png')
bq = pygame.image.load('img/bq.png')
wk = pygame.image.load('img/wk.png')
bk = pygame.image.load('img/bk.png')
begin, end = None, None
eb, ee = None, None
white = True
eng = True
active = False

# render background and buttons
def background(win):
  win.fill((220, 190, 155))
  for x in range(8):
    for y in range(8):
      if (x % 2 == 1 and y % 2 == 0) or (x % 2 == 0 and y % 2 == 1):
        pygame.draw.rect(win, color, (x * square_size, y * square_size, square_size, square_size))
  font = pygame.font.Font(None, 36)
  st = font.render('switch', True, (255, 255, 255))
  sr = st.get_rect(center=(650 + 75, 230 + 25))
  pygame.draw.rect(win, (0, 0, 0), sr.inflate(20, 10))
  win.blit(st, sr)
  rt = font.render('restart', True, (255, 255, 255))
  rr = rt.get_rect(center=(655 + 70, 320 + -5))
  pygame.draw.rect(win, (0, 0, 0), rr.inflate(20, 10))
  win.blit(rt, rr)
  et = 'engine on' if eng else 'engine off'
  es = font.render(et, True, (255, 255, 255))
  er = es.get_rect(center=(655 + 70, 375))
  pygame.draw.rect(win, (0, 0, 0), er.inflate(20, 10))
  win.blit(es, er)

# render chess pieces
def pieces(win, board):
  global eb, ee
  highlight = (255, 255, 0)
  if eb is not None and ee is not None:
    squares = [eb, ee]
    for square in squares:
      x, y = chess.square_rank(square), chess.square_file(square)
      if not white:
        x, y = 7 - x, 7 - y
      pygame.draw.rect(win, highlight, (y * square_size, (7-x) * square_size, square_size, square_size), 5)
  mp = board.piece_map()
  for square, piece in mp.items():
    if dragging and square == begin:
      continue
    x, y = divmod(square, 8)
    if not white:
      x, y = 7-x, 7-y
    img = pimg(piece.symbol())
    if img:
      win.blit(img, (y * square_size, (7-x) * square_size))

# map piece symbols to images
def pimg(symbol):
  return {
    'p': bp, 'n': bn, 'b': bb, 'r': br, 'q': bq, 'k': bk,
    'P': wp, 'N': wn, 'B': wb, 'R': wr, 'Q': wq, 'K': wk,
  }.get(symbol, None)

# convert coordinates to square
def coordinates(coords):
  x, y = coords[0] // square_size, coords[1] // square_size
  if not (0 <= x < 8 and 0 <= y < 8):
    return None
  if not white:
    x, y = 7-x, 7-y
  return chess.square(x, 7-y)

# restart button click detection
def rbtn(pos):
  x, y = pos
  return (725 - 50 <= x <= 725 + 50) and (315 - 15 <= y <= 315 + 15)

# switch button click detection
def sbtn(pos):
  x, y = pos
  return (725 - 50 <= x <= 725 + 50) and (255 - 15 <= y <= 255 + 15)

# engine button click detection
def ebtn(pos):
  x, y = pos
  return (725 - 100 <= x <= 725 + 100) and (365 - 15 <= y <= 365 + 15)

# display pawn promotion choices and return selection
def promotion(color):
  options = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
  images = [wq, wr, wb, wn] if color == chess.WHITE else [bq, br, bb, bn]
  rects = []
  sx = width // 2 - 2 * square_size
  sy = height // 2 - square_size // 2
  for i, img in enumerate(images):
    x = sx + i * square_size
    rect = pygame.Rect(x, sy, square_size, square_size)
    rects.append(rect)
    win.blit(img, rect.topleft)
  pygame.display.flip()
  while True:
    for event in pygame.event.get():
      if event.type == pygame.MOUSEBUTTONDOWN:
        pos = pygame.mouse.get_pos()
        for i, rect in enumerate(rects):
          if rect.collidepoint(pos):
            return options[i]

# engine move
def em(board):
  global active, eb, ee
  def move():
    nonlocal board
    global active, eb, ee
    active = True
    if eng:
      turn = (board.turn == chess.WHITE and not white) or (board.turn == chess.BLACK and white)
      if turn:
        copy = board.copy()
        em = engine.root(engine.depth, copy, True)
        if em:
          with lock:
            print('engine:', em)
            eb, ee = em.from_square, em.to_square
            board.push(em)
            pygame.event.post(pygame.event.Event(pygame.USEREVENT))
    active = False
  thread = threading.Thread(target=move)
  thread.start()

# event handling
def main():
  global begin, end, white, dragging, eng, active, eb, ee
  board = chess.Board()
  running = True
  dragging = False
  drag = None
  dp = (0, 0)
  active = False
  clock = pygame.time.Clock()
  while running:
    for event in pygame.event.get():
      if event.type == QUIT:
        running = False
      elif event.type == MOUSEBUTTONDOWN:
        pos = pygame.mouse.get_pos()
        if rbtn(pos):
          board.reset()
          begin, end = None, None
          eb, ee = None, None
          dragging = False
        elif sbtn(pos):
          white = not white
          begin, end = None, None
          eb, ee = None, None
          dragging = False
          if not active:
            em(board)
        elif ebtn(pos):
          eng = not eng
        else:
          if not active:
            selected = coordinates(pos)
            if selected is not None and board.piece_at(selected):
              begin = selected
              dragging = True
              symbol = board.piece_at(begin).symbol()
              drag = pimg(symbol)
              dp = (pos[0] - square_size // 2, pos[1] - square_size // 2)
      elif event.type == MOUSEBUTTONUP and dragging:
        pos = pygame.mouse.get_pos()
        end = coordinates(pos)
        move = chess.Move(begin, end)
        if board.piece_at(begin) and board.piece_at(begin).piece_type == chess.PAWN:
          if (end in chess.SquareSet(chess.BB_RANK_8) and board.piece_at(begin).color == chess.WHITE) or \
             (end in chess.SquareSet(chess.BB_RANK_1) and board.piece_at(begin).color == chess.BLACK):
            piece = promotion(board.piece_at(begin).color)
            move = chess.Move(begin, end, promotion=piece)
        if move in board.legal_moves:
          board.push(move)
          print('player:', move.uci())
          eb, ee = None, None
          if not active:
            em(board)
        if board.is_game_over():
          board.reset()
          eb, ee = None, None
        begin, end = None, None
        dragging = False
        drag = None
      elif event.type == MOUSEMOTION and dragging:
        dp = (event.pos[0] - square_size // 2, event.pos[1] - square_size // 2)
    background(win)
    pieces(win, board)
    if dragging and drag:
      win.blit(drag, dp)
    pygame.display.flip()
    clock.tick(60)
  pygame.quit()

if __name__ == '__main__':
    main()
