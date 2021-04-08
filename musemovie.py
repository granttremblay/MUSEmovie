#!/usr/bin/env python

import os
import glob

import argparse

import warnings

from astropy.io import fits
from astropy.wcs import WCS

import numpy as np

from tqdm import tqdm as progressbar

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm

import imageio

# Some things we'll be doing throw runtimewarnings that we won't care about.
warnings.filterwarnings('ignore')


def makeMovie(cube, redshift, center, name, thresh=None, frames=30, scalefactor=3.0, vmin=None, vmax=None, contsub=False, whitebg=False, linear=False):
    '''Make the movie'''

    # Read the data cube
    hdulist = fits.open(cube)

    data = hdulist[0].data
    header = hdulist[0].header

    # Most pipeline cubes will have data in 1st extension.
    if data is None:
        data = hdulist[1].data
        header = hdulist[1].header

    hdulist.close()

    number_of_channels = len(data[:, 0, 0])

    # Create the wavelength array
    wavelength = ((np.arange(number_of_channels) + 1.0) -
                  header['CRPIX3']) * header['CD3_3'] + header['CRVAL3']

    # This quick one liner finds the element in the wavelength array
    # that is closest to the "target" wavelength, then returns that element's
    # index.

    # It finds the deviation between each array element and the target value,
    # takes its absolute value, and then returns the index of
    # the element with the smallest value in the resulting array.
    # This is the number that is closest to the target.

    center_channel = (np.abs(wavelength - center)).argmin()
    #print("Emission line centroid for {} is in channel {}".format(name,center_channel))

    movie_start = center_channel - frames
    movie_end = center_channel + frames

    slices_of_interest = np.arange(movie_start, movie_end, 1)

    # Create and scrub the temporary framestore directory
    temp_movie_dir = "framestore/"
    if not os.path.exists(temp_movie_dir):
        os.makedirs(temp_movie_dir)
        print("Created a temporary directory called '{}', where movie frame .png files are stored. You can delete this afterward if you'd like.".format(temp_movie_dir))

    png_files = []

    # Clean the temporary movie directory first
    # If you don't remove all "old" movie frames, the gif will be messed up.
    for f in glob.glob(temp_movie_dir + "*.png"):
        os.remove(f)

    print("Making movie for {} at z={}. Line centroid is in channel {}".format(
        name, round(redshift, 3), center_channel))

    for i, slice in enumerate(progressbar(slices_of_interest)):
        # Perform a dumb continuum subtraction.
        # Risky if you land on another line.
        if contsub is True:
            final_image_data = data[slice, :, :] - \
                data[center_channel - 200, :, :]
        elif contsub is False:
            final_image_data = data[slice, :, :]

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

        # cmap = sns.cubehelix_palette(20, light=0.95, dark=0.15, as_cmap=True)
        cmap = cm.plasma
        if whitebg is True:
            cmap.set_bad('white', 1)
        else:
            cmap.set_bad('black', 1)

        if linear is True:
            ax.imshow(final_image_data, origin='lower', vmin=vmin,
                      vmax=vmax, cmap=cmap, interpolation='None')
        elif linear is False:
            ax.imshow(final_image_data, origin='lower',
                      norm=LogNorm(), vmin=vmin, vmax=vmax, cmap=cmap, interpolation='None')

        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)

        fig.savefig(temp_movie_dir + '{}'.format(i) + '.png', dpi=height)
        png_files.append(temp_movie_dir + '{}'.format(i) + '.png')
        plt.close(fig)

    # Create and scrub the GIF directory
    gif_output_dir = "movies/"
    if not os.path.exists(gif_output_dir):
        os.makedirs(gif_output_dir)
        print("Saving output movies to '{}'.".format(gif_output_dir))

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

    parser.add_argument(
        'cube', help="Name of or full path to a MUSE datacube.")
    parser.add_argument('-z', '--redshift',
                        help="Redshift of the object", default=0, type=float)
    parser.add_argument(
        '-r', '--restwav', help="Rest wavelength of the emission line", default=6563.0, type=float)
    parser.add_argument('-n', '--name', help="Target name",
                        type=str, default=None)
    parser.add_argument('-t', '--thresh', default=None, type=float)
    parser.add_argument(
        '-f', '--frames', help="Number of frames in your movie", default=30, type=int)
    parser.add_argument('-s', '--scalefactor',
                        help="Scale factor for GIF DPI", default=3.0, type=float)
    parser.add_argument('--contsub', help="Perform continuum subtraction?",
                        default=False, action='store_true')
    parser.add_argument('--white', help="Set a white background instead?",
                        default=False, action='store_true')
    parser.add_argument('--linear', help="The scaling is LogNorm by default. Set to linear instead?",
                        default=False, action='store_true')

    parser.add_argument('--vmin', help="Minimum pixel value for color bar",
                        type=float, default=None)

    parser.add_argument('--vmax', help="Maximum pixel value for color bar",
                        type=float, default=None)

    args = parser.parse_args()

    cube = args.cube
    redshift = args.redshift
    restwav = args.restwav
    name = args.name
    thresh = args.thresh
    frames = args.frames
    scalefactor = args.scalefactor
    contsub = args.contsub
    whitebg = args.white
    linear = args.linear

    center = restwav * (1 + redshift)

    makeMovie(cube, redshift, center, name, thresh=thresh,
              frames=frames, scalefactor=scalefactor, vmin=args.vmin, vmax=args.vmax, contsub=contsub, whitebg=whitebg, linear=linear)


if __name__ == '__main__':
    main()
