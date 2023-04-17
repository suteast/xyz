import random
from PIL import Image
import time
a=0
# 迷路のサイズ
maze_size = 50

# 迷路を初期化する
maze = [[0 for y in range(maze_size)] for x in range(maze_size)]

# スタート地点をランダムに選択する
start_x = random.randint(1, maze_size - 2)
start_y = random.randint(1, maze_size - 2)

# スタート地点を壁とする
maze[start_x][start_y] = 1

# 迷路の生成の最大反復回数
max_iterations = 100000

# 迷路を生成する
iteration = 0
while iteration < max_iterations:
    iteration += 1
    
    # 進行方向をランダムに選択する
    direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

    # 次のマスの座標を計算する
    next_x = start_x + direction[0]
    next_y = start_y + direction[1]

    # 次のマスが迷路の範囲内かどうかをチェックする
    if next_x < 1 or next_x >= maze_size - 1 or next_y < 1 or next_y >= maze_size - 1:
        continue

    # 次のマスがまだ通っていない場合、スタート地点と次のマスの間の壁を壊す
    if maze[next_x][next_y] == 0:
        maze[start_x + direction[0] // 2][start_y + direction[1] // 2] = 1
        maze[next_x][next_y] = 1
        start_x, start_y = next_x, next_y

    # 迷路が完成したかどうかをチェックする
    complete = True
    for x in range(1, maze_size - 1):
        for y in range(1, maze_size - 1):
            if maze[x][y] == 0:
                count = 0
                if maze[x + 1][y] == 1:
                    count += 1
                if maze[x - 1][y] == 1:
                    count += 1
                if maze[x][y + 1] == 1:
                    count += 1
                if maze[x][y - 1] == 1:
                    count += 1
                if count == 1:
                    complete = False
    if complete:
        break

# 迷路を画像に変換する
image_size = 10
image = Image.new("RGB", (maze_size * image_size, maze_size * image_size), (255, 255, 255))

for x in range(maze_size):
    for y in range(maze_size):
        if maze[x][y] == 1:
            image.paste((0, 0, 0), (x * image_size, y * image_size, (x + 1) * image_size, (y + 1) * image_size))

image.show()







