import os
import glob
import time

import warnings

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy import coordinates

from astroquery.ned import Ned

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm

import imageio

# Some things we'll be doing throw runtimewarnings that we won't care about.
warnings.filterwarnings('ignore')

def main():

    muse_data_directory = '/Users/grant/Storage/Data/MUSE/Hamer/'
    movie_working_directory = '/Users/grant/Dropbox/SnowClusterMovies/Hamer/'
    line_restwav = 6563 # In Angstroms
    scalefactor = 2.0
    numframes=30

    name_dictionary, coordinate_dictionary = construct_filename_dictionaries(muse_data_directory)

    redshift_dictionary = query_ned_for_redshifts(name_dictionary, coordinate_dictionary)

    emission_line_center_dictionary = map_linecenters(redshift_dictionary, line_restwav)

    # The Name dictionary may have values MISSING from the Redshift Dictionary!
    # You therefore need to iterate on the INTERSECTION of these dictionaries!

    for cube, name in name_dictionary.items():
        if name in redshift_dictionary:
        	makeMovie(movie_working_directory, 
                          cube, 
                          name, 
                          redshift_dictionary[name], 
                          emission_line_center_dictionary[name], 
                          numframes=numframes, 
                          scalefactor=scalefactor,
                          thresh=25.0,
                          cmap=cm.plasma,
                          cmap_nancolor='black', 
                          logscale=True, 
                          contsub=True
                          )
        else:
            print("Skipping movie for {}, it still needs a redshift".format(name))


def map_linecenters(redshift_dictionary, line_restwav):

    # Instantiate the dictionary
    emission_line_center_dictionary = {}

    for name, z in redshift_dictionary.items():
        emission_line_center_dictionary["{}".format(name)] = line_restwav * (1 + z)

    print("Desired emission line ({} Angstroms) redshifted for all targets.".format(line_restwav))

    return emission_line_center_dictionary

def construct_filename_dictionaries(muse_data_directory):
    '''Map ESO archive filenames to target names'''

    print("\n\n ===== MAPPING FILENAMES TO TARGET NAMES ====\n")
    # Create a simple list of the fits filenames
    filelist = glob.glob(muse_data_directory + "*.fits")
    print("MUSE directory set to {}".format(muse_data_directory))

    # Instantiate a dictionary we'll use to map filenames to target names
    name_dictionary = {}
    coordinate_dictionary = {}

    # If these substrings are found in target_name,
    # it won't be a real science datacube, so we're gonna skip that file.
    red_flags = ["(white)", "SKY_"]
    skipped_files = []

    # Loop through the cubelist, skipping white light 2D images
    for fitsfile in filelist:
        hdr = fits.getheader(fitsfile)
        target_name = hdr['OBJECT']

        name_corrections = {"Centaurus": "NGC 4696",
                            "Hydra": "Hydra A", 
                            "R0338": "RX J0338.6+0958"}

        if target_name in name_corrections:
            corrected_target_name = name_corrections[target_name]
            print("Renaming {} to {}".format(target_name, corrected_target_name))
            target_name = corrected_target_name

        ra = hdr['RA']
        dec = hdr['DEC']

        if any(flag in target_name for flag in red_flags):
            skipped_files.append(fitsfile.split("/")[-1])
        else:
            name_dictionary["{}".format(fitsfile)] = target_name
            coordinate_dictionary["{}".format(target_name)] = coordinates.SkyCoord(ra=ra, dec=dec, frame='fk5', unit=(u.deg, u.deg))
            print("{} is {}".format(fitsfile.split("/")[-1], target_name))

    print("Skipped {} files because they were WHITELIGHT or SKY images.".format(len(skipped_files)))

    return name_dictionary, coordinate_dictionary

def query_ned_for_redshifts(name_dictionary, coordinate_dictionary):
    '''Query NED for redshifts based on target names'''

    print("\n\n =========== FINDING REDSHIFTS ========\n")
    # Instantiate an empty dictionary for redshifts
    redshift_dictionary = {}
    continued_failures = []

    target_names = name_dictionary.values()

    for name in target_names:
        try:
            z = Ned.query_object(name)["Redshift"][0]
            redshift_dictionary["{}".format(name)] = z
            print("The NED redshift for {} is {}.".format(name, z))
        except:
            print("Cannot resolve redshift using NAME {}, trying coordinate search.".format(name))
            z = Ned.query_region(coordinate_dictionary[name], radius=20 * u.arcsec, equinox='J2000.0')["Redshift"][0]
            if z == "--":
                continued_failures.append(name)
                print("Still cannot find a redshift for {}, skipping it".format(name))
            elif z < 1.0: # none of these sources are high redshift, this is a dumb sanity check:
                redshift_dictionary["{}".format(name)] = z
                print("{} is at RA={}, Dec={}. NED finds a redshift of {}.".format(name, coordinate_dictionary[name].ra, coordinate_dictionary[name].dec, z))

    if len(continued_failures) > 0:
        print("You need to manually fix these, which still cannot be resolved: ", continued_failures)
        print("In the meantime, they'll be skipped by the movie maker.")
    elif len(continued_failures) == 0:
        print("It SEEMS like all redshifts have successfully been found, ")
        print("but CHECK THESE MANUALLY nonetheless! ")

    print("                  NAME = Z")
    for name, z in redshift_dictionary.items():
        print("               {} = {}".format(name, z))

    return redshift_dictionary



def makeMovie(workingdir, cube, name, redshift, center, numframes=30, scalefactor=2.0, cmap=cm.magma, thresh=None, cmap_nancolor='black', logscale=False, contsub=False):
    '''Make the movie'''

    ########### READ THE DATA CUBE ####################
    hdulist = fits.open(cube)
    data = hdulist[1].data
    header = hdulist[1].header
    wcs_3d = WCS(header)
    #wcs = wcs_3d
    wcs = wcs_3d.dropaxis(2)
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

    movie_start = center_channel - numframes
    movie_end = center_channel + numframes

    slices_of_interest = np.arange(movie_start, movie_end, 1)

    ########### CREATE AND SCRUB THE TEMPORARY FRAMESTORE ##############
    temp_movie_dir = workingdir + "framestore/"
    if not os.path.exists(temp_movie_dir):
        os.makedirs(temp_movie_dir)
        print("Created a temporary directory called '{}', where movie frame .png files are stored. You can delete this afterward if you'd like.".format(temp_movie_dir))

    png_files = []

    # Clean the temporary movie directory first
    # If you don't remove all "old" movie frames, your gif is gonna be messed up.
    for f in glob.glob(temp_movie_dir + "*.png"):
        os.remove(f)
    #####################################################################

    print("\nMaking movie for {} at z={}. Line centroid is in channel {}.".format(
        name, round(redshift, 3), center_channel))

    for i, slice in enumerate(slices_of_interest):

        if contsub is True:
            # Perform a dumb continuum subtraction. Risky if you land on another line.
            cont_sub_image = data[slice, :, :] - data[center_channel - 200, :, :]
            cont_sub_image[cont_sub_image < 0.005] = np.nan
            image = cont_sub_image
        elif contsub is False:
            image = data[slice, :, :]

        if thresh is not None:
            image[image < thresh] = np.nan

        sizes = np.shape(image)
        height = float(sizes[0]) * scalefactor
        width = float(sizes[1]) * scalefactor

        fig = plt.figure()
        fig.set_size_inches(width / height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        #cmap = sns.cubehelix_palette(20, light=0.95, dark=0.15, as_cmap=True)
        cmap.set_bad(cmap_nancolor, 1)

        if logscale is True:
            ax.imshow(image, origin='lower', norm=LogNorm(),
                  cmap=cmap, interpolation='None')
        elif logscale is False:
            ax.imshow(image, origin='lower', cmap=cmap, interpolation='None')

        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)

        fig.savefig(temp_movie_dir + '{}'.format(i) + '.png', dpi=height)
        png_files.append(temp_movie_dir + '{}'.format(i) + '.png')
        plt.close(fig)

    ########### CREATE AND SCRUB THE GIF DIRECTORY ##############
    gif_output_dir = workingdir + "movies/"
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
    print("Done. Saved to {}.".format(gif_name))


if __name__ == '__main__':
    start_time = time.time()
    main()
    runtime = round((time.time() - start_time), 3)
    print("Finished in {} seconds".format(runtime))
