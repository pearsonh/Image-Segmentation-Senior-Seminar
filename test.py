from thresholding import *
from PIL import Image
import numpy as np
from loadBerkeley import loadSegmentation
from bde import bde

ours = baseline_thresholding(Image.open("22093.jpg"))
truth = loadSegmentation("22093.seg")*20
print(bde(ours,truth))
