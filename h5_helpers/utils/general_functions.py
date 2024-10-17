import re
from numpy import asarray, argmin, newaxis, cumsum, insert, pi
import os


def remove_numbers(inp):
    """
    Remove numbers from string (keep only letters)

    :param inp: string input
    """

    return "".join(re.findall("[a-zA-z]+", inp))


def make_utf8(inp):
    """
    Convert input to utf8 instead of bytes

    :param inp: string input
    """

    try:
        inp = inp.decode('utf8')
        return inp
    except (UnicodeDecodeError, AttributeError):
        return inp


def find_closest_indices(arr1, arr2):
    """
    Index mapping between two arrays where each index in arr1 corresponds to the index of the closest value in arr2.

    Parameters:
        arr1 (np.array): The first array.
        arr2 (np.array): The second array, where we find the closest value to each element of arr1.

    Returns:
        np.array: An array of indices from arr2, corresponding to each element in arr1.
    """
    # Convert lists to NumPy arrays if not already
    arr1 = asarray(arr1)
    arr2 = asarray(arr2)

    # Calculate the absolute differences between each element of arr1 and all elements in arr2
    # The resulting matrix will have shape (len(arr1), len(arr2))
    diff_matrix = abs(arr1[:, newaxis] - arr2)

    # Find the index of the minimum value in arr2 for each element in arr1
    # np.argmin will return the indices of the closest values along axis 1 (across columns)
    closest_indices = argmin(diff_matrix, axis=1)

    return closest_indices


def has_integer(input_string):
    """
    Check if string has integer

    :param input: input string

    :return: return boolean (True/False) if integer in string
    """

    if not isinstance(input_string, str):
        input_string = str(input_string)

    return any(char.isdigit() for char in input_string)


def repack(h5):
    """Repack function"""
    print(f'Repack {h5}')
    os.system(f'mv {h5} {h5}.tmp && h5repack {h5}.tmp {h5} && rm {h5}.tmp')


def running_mean(nparray, avgfactor):
    """
    Running mean over numpy axis

    :param nparray: numpy array
    :param avgfactor: averaging factor

    :return: running mean
    """
    cs = cumsum(insert(nparray, 0, 0))

    return (cs[avgfactor:] - cs[:-avgfactor]) / float(avgfactor)


def _degree_to_radian(d):
    """
    Convert degree to radio

    :param d: value in degrees

    :return: return value in radian
    """

    return pi * d / 180