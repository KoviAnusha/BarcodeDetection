from skimage.feature import local_binary_pattern
from skimage.feature import hog
from sklearn.externals import joblib
import glob
import os
pos_feat_ph="../detect/data/featu/pos"
neg_feat_ph="../detect/data/featu/neg"
pospath="../detect/data/dataset/barData/pos/"
negpath="../detect/data/dataset/barData/neg/"


if __name__ == "__main__":
    pos_im_path = pospath
    neg_im_path = negpath
    des_type = "HOG"
    # If feature directories don't exist, create them
    if not os.path.isdir(pos_feat_ph):
        os.makedirs(pos_feat_ph)
    # If feature directories don't exist, create them
    if not os.path.isdir(neg_feat_ph):
        os.makedirs(neg_feat_ph)
    print "Calculating the descriptors for the positive samples and saving them"
    for im_path in glob.glob(os.path.join(pos_im_path, "*")):
        im = imread(im_path, as_gray=True)
        if des_type == "HOG":
            fd = hog(im, orientations=9, pixels_per_cell=(8,8), cells_per_block=(3,3), visualize=False, block_norm="L2-Hys", multichannel=None)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(pos_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
    print "Positive features saved in {}".format(pos_feat_ph)
    print "Calculating the descriptors for the negative samples and saving them"
    for im_path in glob.glob(os.path.join(neg_im_path, "*")):
        im = imread(im_path, as_gray=True)
        if des_type == "HOG":
            fd = hog(im, orientations=9, pixels_per_cell=(8,8), cells_per_block=(3,3), visualize=False, block_norm="L2-Hys", multichannel=None)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(neg_feat_ph, fd_name)
        joblib.dump(fd, fd_path)
    print "Negative features saved in {}".format(neg_feat_ph)
    print "Completed calculating features from training images"
