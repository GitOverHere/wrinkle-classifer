from PIL import Image
import random


# PIL accesses images in Cartesian co-ordinates, so it is Image[columns, rows]
img = Image.new( 'RGB', (250,250), "black") # create a new black image
pixels = img.load() # create the pixel map

for i in range(img.size[0]):    # for every col:
    for j in range(img.size[1]):    # For every row
        pixels[i,j] = (round(random.uniform(0,1)*255), round(random.uniform(0,1)*255), round(random.uniform(0,1)*255)) # set the colour accordingly

img.show()