import argparse, math, os, sys

import numpy as np
import matplotlib.pyplot as plt

from tkinter import *
from PIL import Image, ImageDraw
from keras.models import load_model

LENS = {
    'digi' : [
        '../../models/lens-digi.h5', 
        {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'
        }
    ],
    'alpha' : [
        '../../models/lens-alpha.h5',
        {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 
            13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 
            25: 'Z', 26: 'a', 27: 'b', 28: 'd', 29: 'e', 30: 'f', 31: 'g', 32: 'h', 33: 'n', 34: 'q', 35: 'r', 36: 't'
        }
    ],
    'alnum': [
        '../../models/lens-alnum.h5',
        {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 
            13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 
            25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a',
            37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
        }
    ],
    'kdigi': [
        '../../models/lens-kdigi.h5', 
        {
            0: '೦', 1: '೧', 2: '೨', 3: '೩', 4: '೪', 5: '೫', 6: '೬', 7: '೭', 8: '೮', 9: '೯'
        }
    ],
    'ddigi': [
        '../../models/lens-kdigi.h5', 
        {
            0: '०', 1: '१', 2: '२', 3: '३', 4: '४', 5: '५', 6: '६', 7: '७', 8: '८', 9: '९'
        } 
    ],
    'maths': [
        '../../models/lens-maths.h5', 
        {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '(', 11: ')', 12: '+', 
            13: '/',  14: '=', 15: '*', 16: '-'
        }
    ]
}

os.system('cls')

parser = argparse.ArgumentParser(description='Scratchpad Application to play around with the lenses')
required_args = parser.add_argument_group('required arguments')
required_args.add_argument('-l', '--lens', help='name of the lens', required=True)
args = parser.parse_args()

try:
    model_path, symbols = LENS[args.lens]
    model = load_model(model_path)
except KeyError:
    print('Invalid Lens')
    print('Available Lenses:')
    print(*LENS.keys(), sep='\n')
    sys.exit(0)

os.system('cls')

# Constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CANVAS_WIDTH = 200
CANVAS_HEIGHT = 200

# Global Variables
xpoints=[]
ypoints=[]
x2points=[]
y2points=[]
LABEL_TEXT = ""

window = Tk()
window.title("Scratch Pad")
window.geometry('270x250')

image1 = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT),WHITE)
draw = ImageDraw.Draw(image1)

lbl = Label(window, text='',font=('Arial Bold',20))
lbl.grid(column=1, row=1, columnspan=2)

def paint( event ):
    x1, y1 = ( event.x - 4 ), ( event.y - 4 )
    x2, y2 = ( event.x + 4 ), ( event.y + 4 )
    w.create_oval( x1, y1, x2, y2, fill = 'black' )
    xpoints.append(x1)
    ypoints.append(y1)
    x2points.append(x2) 
    y2points.append(y2)    
    
def imagen ():
    global xpoints
    global ypoints    
    global x2points
    global y2points
    image1 = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT),BLACK)
    draw = ImageDraw.Draw(image1) 
    elementos=len(xpoints)
    for p in range (elementos):
        x=xpoints[p]
        y=ypoints[p]
        x2=x2points[p]
        y2=y2points[p] 
        draw.ellipse((x,y,x2,y2),'white')
        w.create_oval( x-4, y-4, x2+4, y2+4,outline='gray85', fill = 'gray85' )
    size=(28,28)
    image1 = image1.resize(size)
    image1 = image1.convert('L')
    image1 = np.array(image1)
    image1 = image1.astype('float32')
    image1 = 255 - image1
    image1 /= 255.0
    image1 = image1.reshape(-1, 28, 28, 1)
    print(model.predict_classes(image1)[0])
    pred = symbols[model.predict_classes(image1)[0]]
    global LABEL_TEXT
    LABEL_TEXT += str(pred)
    lbl.config(text=LABEL_TEXT)
    print('String', LABEL_TEXT)
    xpoints=[]
    ypoints=[]
    x2points=[]
    y2points=[]

w = Canvas(window, 
           width=CANVAS_WIDTH, 
           height=CANVAS_HEIGHT,bg='gray85')
w.grid(column=1,row=2, rowspan=3)


def delete ():
    global LABEL_TEXT
    LABEL_TEXT = LABEL_TEXT[:-1]
    lbl.config(text=LABEL_TEXT)
    print('String', LABEL_TEXT)
    
def reset():
    global LABEL_TEXT
    LABEL_TEXT = ''
    lbl.config(text=LABEL_TEXT)
    print('String', LABEL_TEXT)
    
w.bind( "<B1-Motion>", paint )

btn_sv = Button(window, text='save', width=5, command=imagen)
btn_sv.grid(column=2,row=2)

btn_dlt = Button(window, text='<-', width=5, command=delete)
btn_dlt.grid(column=2,row=3)

btn_rst = Button(window, text='reset', width=5, command=reset)
btn_rst.grid(column=2,row=4)

window.mainloop()