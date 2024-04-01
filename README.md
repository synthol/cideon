# Cideon

![cideon](https://github.com/synthol/cideon/assets/36903616/5014d8ff-1fc3-4ae5-90cb-54ddb61265a4)

Cideon is a neural network chess engine rated around 1100. It evaluates positions and determines the best move in given situations.

Install
----
```
1. pip install torch numpy pygame python-chess
2. git clone https://github.com/synthol/cideon
```

Play
----
```
1. cd cideon
2. py play.py
```

Train
----
Cideon is trained on 4M games from [Lichess](https://database.lichess.org/).
```
1. train from scratch by placing pgn in directory
2. cd cideon
3. py train.py
```
