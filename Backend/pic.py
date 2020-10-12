#!/usr/bin/env python3
'''''''''''''''''''''''
COPYLEFT LESTERRRY, 2020
'''''''''''''''''''''''
#This script generates picture, just like a VK wall post.
#There's no neural networking here at all
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import time
import sys
import textwrap

def draw(txt):
   wrapper = textwrap.TextWrapper(width=80)
   txt = wrapper.fill(txt)
   img = Image.open("dummy.jpg")
   draw = ImageDraw.Draw(img)
   # font = ImageFont.truetype(<font-file>, <font-size>)
   font = ImageFont.truetype("Verdana.ttf", 23)
   # draw.text((x, y),"Sample Text",(r,g,b))
   draw.text((55, 155),txt,(255,255,255),font=font)
   img.save(f'/var/www/html/neural/pics/{int(time.time())}.jpg')

text = ' '.join(sys.argv[1:])
draw(text)
