from PIL import Image
import numpy as np

def naive_watershed(image):
    #Creates a greyscale version of the input image
    imageGrey = image.convert(mode="L")
    width, height = imageGrey.size
    clumps = []
    accountedFor = []
    for x in range(width):
        for y in range(height):
            if (x,y) in accountedFor:
                continue
            newclump = {"pixels":[(x,y)],"borders":[],"value":imageGrey.getPixel((x,y))}
            print(imageGrey.getPixel((x,y)))
            maybes = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
            accountedFor.append((x,y))
            while len(maybes) > 0:
                cand = maybes.pop()
                if (cand[0] < 0) or (cand[0] == x) or (cand[1] < 0) or (cand[1] == y):
                    continue
                if cand in newclump["pixels"]:
                    continue
                if imageGrey.getPixel(cand) == newclump["value"]:
                    newclump["pixels"].append(cand)
                    maybes.append((cand[0]+1,cand[1]))
                    maybes.append((cand[0]-1,cand[1]))
                    maybes.append((cand[0],cand[1]+1))
                    maybes.append((cand[0],cand[1]-1))
                    accountedFor.append(cand)
                else:
                    newclump["borders"].append(cand)
            clumps.append(newclump)
    pools = []
    walls = []
    for val in range(256):
        for clump in clumps:
            if clump["value"] == val
                borderpools = []
                for pool in pools:
                    for border in pool["boarders"]:
                        if border in clump["pixels"]:
                            borderpools.append(pool)
                if len(borderpools) == 0:
                    #make new pool
                elif len(borderpools) == 1:
                    #add to pool
                else:
                    for pixel in clump["pixels"]:
                        walls.append(pixel)
    #make new image and make the walls black and other stuff white

if __name__ == "__main__":
    img = Image.open("22093.jpg")
    img = Image.fromarray(baseline_thresholding(img)*200)
    img.show()
