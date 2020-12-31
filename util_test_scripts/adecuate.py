"""
Call with this command (tiles1/* should be replaced by the folder that has all
the images to adecuate):
ls -rt tiles1/* | xargs -I{} python adecuate.py {}

Tomas Alvarez Vanoli
28/12/2020

"""
from skimage import io, filters, transform
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import sys
img_list=sys.argv[1:]
for img_dir in img_list:
    img_o = rgb2gray(io.imread(img_dir))[10:-10,10:-10]
    thr_li=filters.threshold_li(img_o)
    img1=transform.resize(img_o<=thr_li, [256,512])
    io.imsave(img_dir.replace('.','_p.'),img_as_ubyte(img1))
