import os
import glob

import argparse

import warnings

from astropy.io import fits
from astropy.wcs import WCS

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm

import imageio

# Some things we'll be doing throw runtimewarnings that we won't care about.
warnings.filterwarnings('ignore')


def makeMovie(cube, name, thresh=None, scalefactor=3.0):
    '''Make the movie'''

    ########### READ THE DATA CUBE ####################
    hdulist = fits.open(cube)

    data = hdulist[0].data
    header = hdulist[0].header

    hdulist.close()

    number_of_channels = len(data[0,:, 0, 0])

    # This quick one liner finds the element in the wavelength array
    # that is closest to the "target" wavelength, then returns that element's
    # index.

    # It finds the deviation between each array element and the target value,
    # takes its absolute value, and then returns the index of
    # the element with the smallest value in the resulting array.
    # This is the number that is closest to the target.

    movie_start = 0
    movie_end = number_of_channels

    slices_of_interest = np.arange(movie_start, movie_end, 1)

    ########### CREATE AND SCRUB THE TEMPORARY FRAMESTORE ##############
    temp_movie_dir = "framestore/"
    if not os.path.exists(temp_movie_dir):
        os.makedirs(temp_movie_dir)
        print("Created a temporary directory called '{}', where movie frame .png files are stored. You can delete this afterward if you'd like.".format(temp_movie_dir))

    png_files = []

    # Clean the temporary movie directory first
    # If you don't remove all "old" movie frames, your gif is gonna be messed up.
    for f in glob.glob(temp_movie_dir + "*.png"):
        os.remove(f)
    #####################################################################

    print("Making movie")

    for i, slice in enumerate(slices_of_interest):

        final_image_data = data[0, slice, :, :]

        if thresh is not None:
            final_image_data[final_image_data < thresh] = np.nan

        sizes = np.shape(final_image_data)
        height = float(sizes[0]) * scalefactor
        width = float(sizes[1]) * scalefactor

        fig = plt.figure()
        fig.set_size_inches(width / height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        #cmap = sns.cubehelix_palette(20, light=0.95, dark=0.15, as_cmap=True)
        cmap = cm.viridis
        cmap.set_bad('black', 1)

        ax.imshow(final_image_data, origin='lower', cmap=cmap, interpolation='None')     

        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)

        fig.savefig(temp_movie_dir + '{}'.format(i) + '.png', dpi=height)
        png_files.append(temp_movie_dir + '{}'.format(i) + '.png')
        plt.close(fig)

    ########### CREATE AND SCRUB THE GIF DIRECTORY ##############
    gif_output_dir = "movies/"
    if not os.path.exists(gif_output_dir):
        os.makedirs(gif_output_dir)
        print("Saving output movies to '{}'.".format(gif_output_dir))
    #############################################################

    gif_name = gif_output_dir + '{}.gif'.format(name)
    gif_frames = []

    # Remove any old GIFs you might have made
    if os.path.isfile(gif_name):
        os.remove(gif_name)

    for filename in png_files:
        gif_frames.append(imageio.imread(filename))

    imageio.mimsave(gif_name, gif_frames)
    print("Done. Saving movie to {}.".format(gif_name))


def main():

    parser = argparse.ArgumentParser(description='Make a movie of a MUSE cube')

    parser.add_argument('cube', help="the ALMA cube")

    parser.add_argument('-n', '--name', help="Target name", type=str, default=None)

    parser.add_argument('-t', '--thresh', default=None, type=float)

    args = parser.parse_args()

    cube = args.cube    
    name = args.name
    thresh = args.thresh

    makeMovie(cube, name, thresh)


if __name__ == '__main__':
    main()
