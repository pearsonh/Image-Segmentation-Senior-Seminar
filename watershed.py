from PIL import Image
import numpy as np

def naive_watershed(image):
    #Creates a greyscale version of the input image
    imageGrey = image.convert(mode="L")
    width, height = imageGrey.size
    clumps = []
    accountedFor = []
    print("starting clumping")
    #'''
    for x in range(width):
        print(x)
        for y in range(height):
            if (x,y) in accountedFor:
                continue
            newclump = {"pixels":[(x,y)],"borders":[],"value":imageGrey.getpixel((x,y))}
            #print("START:")
            #print((x,y))
            maybes = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
            accountedFor.append((x,y))
            while len(maybes) > 0:
                cand = maybes.pop()
                #print(cand)
                if (cand[0] < 0) or (cand[0] == width) or (cand[1] < 0) or (cand[1] == height):
                    continue
                if cand in newclump["pixels"]:
                    continue
                if imageGrey.getpixel(cand) == newclump["value"]:
                    newclump["pixels"].append(cand)
                    maybes.append((cand[0]+1,cand[1]))
                    maybes.append((cand[0]-1,cand[1]))
                    maybes.append((cand[0],cand[1]+1))
                    maybes.append((cand[0],cand[1]-1))
                    accountedFor.append(cand)
                else:
                    newclump["borders"].append(cand)
            clumps.append(newclump)

    print("done clumping")
    #print(clumps)
    pools = []#'''
    returnImage = np.zeros(shape=(height,width))
    #'''
    for val in range(256):
        print(val)
        for clump in clumps:
            if clump["value"] == val:
                borderpools = []
                for pool in pools:
                    for border in pool["borders"]:
                        if border in clump["pixels"]:
                            borderpools.append(pool)
                if len(borderpools) == 0:
                    newpool = {"borders":clump["borders"]}
                    pools.append(newpool)
                elif len(borderpools) == 1:
                    pool = borderpools[0]
                    pool["borders"] = [x for x in pool["borders"]+clump["borders"] if x not in clump["pixels"]]
                else:
                    for (x,y) in clump["pixels"]:
                        returnImage[y,x] = 255
    
    print("done segmenting")
    return returnImage

if __name__ == "__main__":
    img = Image.open("22093.jpg")
    img = Image.fromarray(naive_watershed(img))
    img.show()
