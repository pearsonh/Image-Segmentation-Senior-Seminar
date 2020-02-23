  This term, our image segmentation comps group created implementations of two algorithms. After spending a couple of weeks
researching the task of image segmentation, including what it is and how it is used, we decided on creating versions of
the K-means algorithm and the thresholding algorithm. We also decided on two data sets, and created an edge based and
a region based evalutation metric.

  We began our work by creating a working version of a K-means image segmentation algorithm, that takes in an image and an
an amount of segments to produce, and outputs a 2d array that marks which segment each pixel is in. Our other algorithm is
a thresholding algorithm that calculates the average greyscale value over the pixels, and uses this to make each pixel in
one of two segments. Lastly, we built a parser to translate the ground truths in our datasets to the same format as the
output of our functions, so that we may more easily compare our algorithm and get results.

  To run, first run "pip install Pillow"
  To run in the command line, run "python experiment.py" to use our command line tool. The command line tool has a list
of commands you can run:

    * --threshold - Runs thresholding algorithm and displays results
    * --kmeans - Runs K-means algorithm and displays results
    * --segments - Specifies hyperparameter for K-means (default 1)
    * --image - specifies image to segment (default is the included photo)
    * --displayOriginal - puts the original image side by side with the segmentation to be
                          more easily compared
    * --evaluate - currently does nothing, to be implemented when the group begins to finalize the experiment next term
