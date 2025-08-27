import tkinter as tk
from PIL import Image, ImageTk
from time import sleep

height = 800
width = 800

root = tk.Tk()
root.title("Knight's Tour Animation")

frame = tk.Frame(root)
frame.pack(fill = tk.BOTH, expand = True)

speed = tk.StringVar(root)
speed.set("1x")
nl = tk.StringVar(root)
nl.set("5x5")

nn = tk.OptionMenu(frame, nl, "5x5", "6x6", "7x7", "8x8")
nn.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)
w = tk.OptionMenu(frame, speed, "1/4x", "1/2x", "1x", "2x","4x")
w.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

xtxt = tk.Text(frame, height = 1, width =7)
xtxt.pack(side = tk.RIGHT, fill = tk.BOTH, expand = True)

ytxt = tk.Text(frame, height = 1, width =7)
ytxt.pack(side = tk.RIGHT, fill = tk.BOTH, expand = True)

lbl = tk.Label(frame, text = "Please the starting column (left field) and the starting row (right field)")
lbl.pack(side = tk.BOTTOM, fill = tk.BOTH, expand = True)

def starter():
    global sp
    if nl.get() == "5x5":
        no=5
    elif nl.get() == "6x6":
        no=6
    elif nl.get() == "7x7":
        no=7
    elif nl.get() == "8x8":
        no=8
    if speed.get() == "1x":
        sp=0.6
    elif speed.get() == "2x":
        sp=0.3
    elif speed.get() == "4x":
        sp=0.15
    elif speed.get() == "1/2x":
        sp=1.2
    elif speed.get() == "1/4x":
        sp=2.4
    startx = int(xtxt.get(1.0, "end-1c"))
    starty = int(ytxt.get(1.0, "end-1c"))
    if((startx >= 0 and startx < no) and (starty >= 0 and starty < no)):
        solveKT(no, startx, starty)
    else:
        lbl.config(text = "Invalid Starting Position")

start = tk.Button(frame, text="Start", command=starter)
start.pack(side = tk.LEFT, fill = tk.BOTH, expand = True)

canvas = tk.Canvas(root, width=width, height=height)

def isSafe(x, y, board, n):
    if(x >= 0 and y >= 0 and x < n and y < n and board[x][y] == -1):
        return True
    return False


def printSolution(n, board):
    for i in range(n):
        for j in range(n):
            print(board[i][j], end=' ')
        print()
    print(board)


def solveKT(n, start_x, start_y):
    board = [[-1 for i in range(n)]for i in range(n)]

    move_x = [2, 1, -1, -2, -2, -1, 1, 2]
    move_y = [1, 2, 2, 1, -1, -2, -2, -1]

    board[start_x][start_y] = 0

    pos = 1

    if(not solveKTUtil(n, board, start_x, start_y, move_x, move_y, pos)):
        lbl.config(text = "Solution does not exist")
    else:
        printSolution(n, board)
        indecies = []
        for k in range(n**2):
            for row in range(n):
                for col in range(n):
                    if(board[row][col]==k):
                        indecies.append([row, col])

        print(indecies)
        canvas_board(indecies, n)

def solveKTUtil(n, board, curr_x, curr_y, move_x, move_y, pos):
    canvas.pack()
    if(pos == n**2):
        return True

    for i in range(8):
        new_x = curr_x + move_x[i]
        new_y = curr_y + move_y[i]
        if(isSafe(new_x, new_y, board, n)):
            board[new_x][new_y] = pos
            if(solveKTUtil(n, board, new_x, new_y, move_x, move_y, pos+1)):
                return True

            board[new_x][new_y] = -1
    return False

def canvas_board(indecies, n):
    global sp

    sheight = height // n
    swidth = width // n

    knight = Image.open("knight.png")
    knight = knight.resize((sheight, swidth), Image.BILINEAR)
    knight = ImageTk.PhotoImage(knight)

    visited = []
    for k in range(n**2):
        canvas.delete("all")
        for i in range(n):
            for j in range(n):
                if(i == indecies[k][0] and j == indecies[k][1]):
                    visited.append([j, i])
                    canvas.create_image(j*swidth, i*sheight, image=knight, anchor=tk.NW)
                    sleep(sp)
                elif(i % 2 == 0 and not (i == indecies[k][0] and j == indecies[k][1])):
                    if(j % 2 == 0):
                        canvas.create_rectangle((j*swidth, i*sheight, j*swidth+swidth, i*sheight+sheight), fill="white")
                    else:
                        canvas.create_rectangle((j*swidth, i*sheight, j*swidth+swidth, i*sheight+sheight), fill="black")
                elif(i % 2 == 1 and not (i == indecies[k][0] and j == indecies[k][1])):
                    if (j % 2 == 0):
                        canvas.create_rectangle((j*swidth, i*sheight, j*swidth+swidth, i*sheight+sheight), fill="black")
                    else:
                        canvas.create_rectangle((j*swidth, i*sheight, j*swidth+swidth, i*sheight+sheight), fill="white")
        for v in range(k):
            canvas.create_text(visited[v][0] * sheight + sheight / 2, visited[v][1] * swidth + swidth / 2, text=v,
                               fill="red", font=('Helvetica 15 bold'), anchor=tk.NW)
        root.update()
    canvas.create_image(indecies[(n**2)][0]* swidth,indecies[(n**2)][1] * sheight, image=knight, anchor=tk.NW)
    root.update()




root.mainloop()



