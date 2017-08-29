#!/usr/bin/env python

'''
musemovie.py: Make a GIF movie of a MUSE cube
'''

__author__ = "Dr. Grant R. Tremblay"

import os
import sys

import numpy as np

import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from astropy.io import fits

import argparse

import imageio


def readcube():
    '''Read the data cube'''

    image_file = 'A2597.final.fits'
    hdulist = fits.open(image_file)
    image_data = hdulist[0].data
    print(image_data.shape)
    hdulist.close()

def makeMovie():

    slices_of_interest = np.arange(1490, 1565, 1)
    png_files = []

    for i, slice in enumerate(slices_of_interest):
        image = image_data[slice,105:265,105:265]
        fig = plt.figure(figsize=(3,3))
        plt.axis('off')
        cmap = matplotlib.cm.magma
        cmap.set_bad('black',1)
        frame = plt.imshow(image, origin='lower', norm=LogNorm(), vmin=0.001, vmax=0.1, cmap='magma', interpolation='None')
        fig.savefig('output_images/' + '{}'.format(i) + '.png', dpi=100, bbox_inches='tight')
        png_files.append('output_images/' + '{}'.format(i) + '.png')

def makeGif():
    gif_frames = []

    for filename in png_files:
        gif_frames.append(imageio.imread(filename))

    imageio.mimsave('out.gif', gif_frames)

def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile', help="Input file", type=argparse.FileType('r'))
    parser.add_argument('-o', '--outfile', help="Output file",
                        default=sys.stdout, type=argparse.FileType('w'))

    args = parser.parse_args(arguments)

    print(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
