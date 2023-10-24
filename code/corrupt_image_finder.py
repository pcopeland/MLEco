import os
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import skimage
from PIL import Image
from CONSTANTS import PATH_TO_IMAGE
from data_collection.vso_search_result import VSOSearchResult
from data_collection.gong_sampler import sample_by_size

# -------------------------------------------------
# Deprecated! See 'halpha_anomaly_detector.py'.
# -------------------------------------------------


def get_each_image(url: str,
                   path: str = os.path.join(PATH_TO_IMAGE, '.image.jpg')):
    """
    This function accesses a passed url, downloads the image, and saves it to
    the local folder for processing. The image is overwritten each time the
    function is called.

    Parameters
    ----------
    url
        a URL address corresponding to a single image
    path
        the path filename to write and read the image file

    Returns
    -------
        the image
    """
    image_url = url
    image_temp = Image.open(requests.get(image_url, stream=True).raw)
    
    # image_temp.save(path)
    # img = Image.open(path)
    
    # return img
    return image_temp


def block_reduce_image(img, side_length: int):
    """
    This function down-samples an image to collect the average pixel values
    and converts them into an array.

    Parameters
    ----------
    img
        the image to be processed
    side_length
        the desired side length of the grid

    Returns
    -------
        a dataframe filled with specified image brightness values
    """
    pixel_array = np.array(img)
    down_sample = int(img.size[0] / side_length)
    array = skimage.measure.block_reduce(pixel_array,
                                         (down_sample, down_sample), np.mean)
    
    return array


def segment_images(urls: list, grid_size: int,
                   path: str = os.path.join(PATH_TO_IMAGE, '.image.jpg')):
    """
    This function calculates the brightness of each resized pixel and stores
    the data in a dataframe for processing.

    Parameters
    ----------
    urls
        an array of image urls for which the average brightness per pixel is
        being calculated
    grid_size
        the desired side length of the grid
    path
        the path filename to write and read the image file

    Returns
    -------
        a dataframe with dimensions len(urls) by grid_size, filled with
        specified image brightness values
    """
    df = pd.DataFrame()

    for i in tqdm(urls):
        bw_img = get_each_image(i).convert("L")
        ds_array = block_reduce_image(bw_img, grid_size)
        
        row = ds_array.flatten()
        row = pd.Series(row)
        row = row.to_frame().T
        
        df = pd.concat([df, row], ignore_index=True)
    
    df['URL'] = urls
    df.set_index('URL', inplace=True)

    # os.remove(path)

    return df


def mask_cells_outside_iqr(df, lower_range=4, upper_range=96):
    """
    This function subtracts the median value of each row from each cell in
    the row and calculates ranges based on preset percentiles that are
    defined as the lower_range and Upper_range. It then calculates a range
    based on a multiple of the "inner quartile range" and replaces each cell
    value with a boolean value indicating whether that value falls within
    (True), or outside of (False), the given range.

    Parameters
    ----------
    df
        a dataframe of pixel brightness values
    lower_range
        the low bound percentile for calculating the acceptable range of
        pixel values
    upper_range
        the upper bound percentile for calculating the acceptable range of
        pixel values

    Returns
    -------
        an updated dataframe
    """
    df = df.sub(df.median(axis=1), axis=0)
    
    for col in df:
        pl = np.percentile(df[col], lower_range)
        ph = np.percentile(df[col], upper_range)
        iqr = ph - pl
        lower = pl - iqr * 1.5
        upper = ph + iqr * 1.5
        df[col] = df[col].between(lower, upper)
    
    return df


def find_corrupt_images(df):
    """
    This function applies a suite of tools to determine which images are
    likely corrupted and unsuitable for further processing.

    Parameters
    ----------
    df
        a data frame of pixel brightness values

    Returns
    -------
        a list of unique image identifications for corrupted images
    """
    df = mask_cells_outside_iqr(df)
    counts = df.apply(pd.Series.value_counts, axis=1)
    
    if False in counts.columns:
        corrupted = counts.index[counts[False] > 0]
    else:
        corrupted = []
    
    return corrupted


if __name__ == "__main__":
    begin = "2012-01-01 00:00:01"
    end = "2012-02-01 23:59:59"
    image_data: VSOSearchResult = sample_by_size(begin, end, 100, "maunaloa")
    print("{} observations have been found.".format(image_data.n_queried_files))
    image_data.generate_url_metadata(fits_urls=True, header_urls=True,
                                     jpg_urls=True)
    image_urls = image_data.jpg_urls
    start = time.time()
    cell_data = segment_images(image_urls, 16)
    corrupt_images = find_corrupt_images(cell_data)
    end = time.time()
    print("{} corrupted images were found: ".format(len(corrupt_images)))
    print(corrupt_images)
    print("____________________________________________")
    print("The execution time is: ", (end - start) / 60, " minutes")
    print("____________________________________________")
