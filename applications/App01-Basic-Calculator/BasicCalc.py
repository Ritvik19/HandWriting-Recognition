from tkinter import *
import math

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from keras.models import load_model

from keras.models import load_model
model = load_model('../../models/lens-digi.h5')

white = (255, 255, 255)
black = (0, 0, 0)

window = Tk()
window.title("Handwriting Calculator")
window.geometry('265x165')

canvas_width = 120
canvas_height = 120
image1 = Image.new("RGB", (canvas_width, canvas_height),white)
draw = ImageDraw.Draw(image1)
xpoints=[]
ypoints=[]
x2points=[]
y2points=[]

LABEL_TEXT = ""

lbl = Label(window, text='',font=('Arial Bold',20))
lbl.grid(column=1, row=1, columnspan=3)

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
    image1 = Image.new("RGB", (canvas_width, canvas_height),black)
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
    pred = model.predict_classes(image1)[0]
    global LABEL_TEXT
    LABEL_TEXT += str(pred)
    lbl.config(text=LABEL_TEXT)
    print('String', LABEL_TEXT)
    xpoints=[]
    ypoints=[]
    x2points=[]
    y2points=[]

w = Canvas(
    window, width=canvas_width, height=canvas_height,bg='gray85'
)
w.grid(column=1,row=2, rowspan=4)


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
    
def fun(op):
    global LABEL_TEXT
    LABEL_TEXT += op
    lbl.config(text=LABEL_TEXT)
    print('String', LABEL_TEXT)

def eql():
    global LABEL_TEXT
    LABEL_TEXT = str(eval(LABEL_TEXT))
    lbl.config(text=LABEL_TEXT)
    print('String', LABEL_TEXT)
    
w.bind( "<B1-Motion>", paint )

btn_add = Button(window, text='+', width=5, command=lambda: fun('+'))
btn_add.grid(column=2,row=2)

btn_sub = Button(window, text='-', width=5, command=lambda: fun('-'))
btn_sub.grid(column=2,row=3)

btn_mul = Button(window, text='*', width=5, command=lambda: fun('*'))
btn_mul.grid(column=2,row=4)

btn_idiv = Button(window, text='//', width=5, command=lambda: fun('//'))
btn_idiv.grid(column=2,row=5)

btn_dot = Button(window, text='.', width=5, command=lambda: fun('.'))
btn_dot.grid(column=3,row=2)

btn_mod = Button(window, text='%', width=5, command=lambda: fun('%'))
btn_mod.grid(column=3,row=3)

btn_pow = Button(window, text='**', width=5, command=lambda: fun('**'))
btn_pow.grid(column=3,row=4)

btn_tdiv = Button(window, text='/', width=5, command=lambda: fun('/'))
btn_tdiv.grid(column=3,row=5)

btn_sv = Button(window, text='save', width=5, command=imagen)
btn_sv.grid(column=4,row=2)

btn_eql = Button(window, text='=', width=5, command=eql)
btn_eql.grid(column=4,row=3)

btn_dlt = Button(window, text='<-', width=5, command=delete)
btn_dlt.grid(column=4,row=4)

btn_rst = Button(window, text='reset', width=5, command=reset)
btn_rst.grid(column=4,row=5)


window.mainloop()