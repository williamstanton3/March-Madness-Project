import pandas as pd
import math
from collections import Counter

from statistics import mean, median, stdev
from typing import Sequence, Callable, Any

import os
import sys
scriptdir = os.path.abspath(os.path.dirname(__file__))
if scriptdir not in sys.path: sys.path.append(scriptdir)
from data_inspector import count_categories

def transform_feature(df: pd.DataFrame, col_name: str, action: str, args: list[Any], kwargs: dict[str,Any]):
    """Transforms a single column of the dataframe using the specified modification

    Positional Arguments:
    df       - The dataframe on which an attribute will be transformed (may be modified in place)
    col_name - The name of the column whos values will be changed
    action   - One of the following function names (defined in this file)
                  1. z_score_norm
                  2. min_max_norm
                  3. make_named_bins
                  4. make_mean_bins
                  5. make_median_bins
                  6. make_min_bins
                  7. make_max_bins
    args     - A list of the positional arguments required by the action function
    kwargs   - A dictionary of the keyword arguments required by the action function
    """
    # identify the correct function to call given the specified action
    if action == 'z_score_norm': func = z_score_norm
    elif action == 'min_max_norm': func = min_max_norm
    elif action == 'merge_uncommon': func = merge_uncommon
    elif action == 'make_named_bins': func = make_named_bins
    elif action == 'make_mean_bins': func = make_mean_bins
    elif action == 'make_median_bins': func = make_median_bins
    elif action == 'make_min_bins': func = make_min_bins
    elif action == 'make_max_bins': func = make_max_bins
    else: raise ValueError(f"Unrecognized transformation action: {action}")
    # apply this function to the specified column
    df[col_name] = func(df[col_name], *args, **kwargs) # type: ignore

def z_score_norm(items: Sequence[int|float]) -> Sequence[float]:
    """Translates all values into standard deviations above and below the mean"""
    my_mean = sum(items) / len(items)
    standard_div = math.sqrt(sum((x - my_mean) ** 2 for x in items) / (len(items)-1))
    
    return [(num - my_mean) / standard_div for num in items]

def min_max_norm(items: Sequence[int|float]) -> Sequence[float]:
    """Scales all items into the range [0, 1]"""
    min_val = min(items)
    max_val = max(items)
    range = max_val - min_val
    
    return [(x - min_val) / range for x in items]

def merge_uncommon(items: Sequence[str], default: str = 'OTHER',
                   max_categories: int|None = None, 
                   min_count: int|None = None, 
                   min_pct: float|None = None) -> Sequence[str]:
    """Merges infrequent categorical labels into a single miscellaneous category
    
    Positional Arguments:
    items   - A sequence of categorical labels to be transformed
    default - The default value with which to replace uncommon labels

    Keyword Arguments:
    max_categories - The maximum number of distinct labels to be kept (keep most common)
    min_count      - The minimum number of examples a label must have to be kept
    min_pct        - The minimum percentage of the dataset a label must represent to be kept

    returns a transformed version of items where uncommon labels are replaced with the default value
    """
    # NOTE: Exactly ONE of the keyword arguments should be specified!
    #       More or less should result in an exception!

    # get the fequencies of the items
    frequencies = Counter(items)

    # check to make sure they entered exactly one keyword argument 
    if max_categories is not None:
        if (min_count is not None) or (min_pct is not None):
            raise Exception("ONLY ONE KEYWORD ARGUMENT ALLOWED!")
    if min_count is not None:
        if (max_categories is not None) or (min_pct is not None):
            raise Exception("ONLY ONE KEYWORD ARGUMENT ALLOWED!")
    if min_pct is not None:
        if (min_count is not None) or (max_categories is not None):
            raise Exception("ONLY ONE KEYWORD ARGUMENT ALLOWED!")

    # case 1 - max_categories is provided
    if (max_categories is not None):
        # find the x most common labels
        most_common_strings = frequencies.most_common(max_categories)
        
        # store a set of the strings to keep (not group into Other)
        dont_group = set()
        for word, count in most_common_strings:
            dont_group.add(word)

    # case 2 - min_count is provided
    elif (min_count is not None):
        # store a set of strings to keep (not group into Other)
        dont_group = set() 

        for label, count in frequencies.items():
            if count >= min_count:
                dont_group.add(label)

    # case 3 - min_pct is provided 
    elif (min_pct is not None):
        # store a set of strings to keep (not group into Other)
        dont_group = set() 
        num_total = len(items) 

        for label, count in frequencies.items():
            percentage = count / num_total 
            if percentage >= min_pct:
                dont_group.add(label)

    # once we know which ones are are not grouping, we can actually group the remainder 
    final_list = []
    for item in items:
        if item in dont_group:
            final_list.append(item)
        else:
            final_list.append(default)

    return final_list


def make_named_bins(items: Sequence[int|float], cut: str, names: Sequence[str]):
    """Bins items using the specified strategy and represents each with one of the given names"""
    # HINT: you should make use of the _find_bins function defined below
    num_bins = len(names)

    # get the bin number for each bin using _find_bins
    bin_nums = _find_bins(items, cut, num_bins)

    # map each number in bin_nums to its corresponding name from names
    named_bins = [] 
    for number in bin_nums:
        named_bins.append(names[number])
    return named_bins

def make_mean_bins(items: Sequence[int|float], cut: str, bin_count: int) -> Sequence[int|float]:
    """Bins items using the specified cut strategy and represents each bin with its mean"""
    # HINT: you should make use of the _find_bins function defined below
    
    # get the bin numbers
    bin_nums = _find_bins(items, cut, bin_count)

    # create an empty list where each index represents a bin
    bins = [] 
    for i in range(bin_count):
        bins.append([])

    # fill in the bins list with the items in each bin (creates list of lists)
    for i in range(len(items)):
        if hasattr(items, 'iloc'):
            item = items.iloc[i]
        else:
            item = items[i] 
        bin_num = bin_nums[i] # the number bin that this item is in
        bins[bin_num].append(item)

    # get the mean of each bin
    bin_means = []
    for bin in bins:
        my_mean = mean(bin)
        bin_means.append(my_mean) 
    
    # return a list the size of items where each item is represented as its bin mean 
    return_list = [] 
    for i in range(len(items)):
        bin_num = bin_nums[i] # get bin num
        bin_mean = bin_means[bin_num] # get bin mean
        return_list.append(bin_mean)
        
    return return_list

def make_median_bins(items: Sequence[int|float], cut: str, bin_count: int) -> Sequence[int|float]:
    """Bins items using the specified cut strategy and represents each bin with its median"""
    # HINT: you should make use of the _find_bins function defined below
    
    # get the bin numbers
    bin_nums = _find_bins(items, cut, bin_count)

    # create an empty list where each index represents a bin
    bins = [] 
    for i in range(bin_count):
        bins.append([])

    # fill in the bins list with the items in each bin (creates list of lists)
    for i in range(len(items)):
        item = items[i] # the actual item from the args
        bin_num = bin_nums[i] # the number bin that this item is in
        bins[bin_num].append(item)

    # get the median of each bin
    bin_meds = []
    for bin in bins:
        med = median(bin)
        bin_meds.append(med) 

    # return a list the size of items where each item is represented as its bin median 
    return_list = [] 
    for i in range(len(items)):
        bin_num = bin_nums[i] # get bin num
        bin_med = bin_meds[bin_num] # get bin median
        return_list.append(bin_med)
        
    return return_list

def make_min_bins(items: Sequence[int|float], cut: str, bin_count: int) -> Sequence[int|float]:
    """Bins items using the specified cut strategy and represents each bin with its minimum value"""
    # HINT: you should make use of the _find_bins function defined below
    
    # get the bin numbers
    bin_nums = _find_bins(items, cut, bin_count)

    # create an empty list where each index represents a bin
    bins = [] 
    for i in range(bin_count):
        bins.append([])

    # fill in the bins list with the items in each bin (creates list of lists)
    for i in range(len(items)):
        item = items[i] # the actual item from the args
        bin_num = bin_nums[i] # the number bin that this item is in
        bins[bin_num].append(item)

    # get the min of each bin
    bin_mins = []
    for bin in bins:
        my_min = min(bin)
        bin_mins.append(my_min) 

    # return a list the size of items where each item is represented as its bin minimum
    return_list = [] 
    for i in range(len(items)):
        bin_num = bin_nums[i] # get bin num
        bin_min = bin_mins[bin_num] # get bin min
        return_list.append(bin_min)
        
    return return_list

def make_max_bins(items: Sequence[int|float], cut: str, bin_count: int) -> Sequence[int|float]:
    """Bins items using the specified cut strategy and represents each bin with its maximum value"""
    # HINT: you should make use of the _find_bins function defined below
    # get the bin numbers
    bin_nums = _find_bins(items, cut, bin_count)

    # create an empty list where each index represents a bin
    bins = [] 
    for i in range(bin_count):
        bins.append([])

    # fill in the bins list with the items in each bin (creates list of lists)
    for i in range(len(items)):
        item = items[i] # the actual item from the args
        bin_num = bin_nums[i] # the number bin that this item is in
        bins[bin_num].append(item)

    # get the max of each bin
    bin_maxs = []
    for bin in bins:
        my_max = max(bin)
        bin_maxs.append(my_max) 

    # return a list the size of items where each item is represented as its bin maximum
    return_list = [] 
    for i in range(len(items)):
        bin_num = bin_nums[i] # get bin num
        bin_max = bin_maxs[bin_num] # get bin max
        return_list.append(bin_max)
        
    return return_list

def _find_bins(items: Sequence[int|float], cut: str, bin_count: int) -> Sequence[int]:
    """Bins the items and returns a sequence of bin numbers in the range [0,bin_count)"""
    # identify the bin cutoffs based on strategy
    if cut == 'width':
        boundaries = _get_equal_width_cuts(items, bin_count)
    elif cut == 'freq':
        boundaries = _get_equal_frequency_cuts(items, bin_count)
    else:
        raise ValueError(f"Unrecognized bin cut strategy: {cut}")
    # determine the bin of each item using those cutoffs and return the list of bins
    return [_find_bin(item, boundaries) for item in items]

def _find_bin(item: int|float, boundaries: list[tuple[float,float]]) -> int:
    """Assigns a given item to one of the bins defined by the given boundaries bin_min <= x < bin_max"""
    # check edge cases outside the range of the bins
    if item < boundaries[0][0]: return 0
    if item >= boundaries[-1][-1]: return len(boundaries)-1
    # otherwise find the correct bin
    for bin_num,(bin_min,bin_max) in enumerate(boundaries):
        if bin_min <= item and item < bin_max:
            return bin_num
    # this point should never be reached so raise an exception
    raise ValueError(f"Unable to place {item} in any of the bins")

def _get_equal_width_cuts(items: Sequence[int|float], bin_count: int) -> list[tuple[float,float]]:
    """Returns a list of the lower and upper cutoffs for each of the equal width bins"""
    # find the minimum and maximum values in items
    low: float = min(items)
    high: float = max(items)
    # define the bin width as 1/bin_count of the difference between the min and max values
    width: float = (high - low) / bin_count
    # compute the bin boundaries using this width
    boundaries: list[tuple[float,float]] = []
    for bin_num in range(bin_count):
        # identify the boundaries for this bin and add them to the list
        bin_min = low + bin_num * width
        bin_max = low + (bin_num+1) * width
        boundaries.append((bin_min, bin_max))
    return boundaries

def _get_equal_frequency_cuts(items: Sequence[int|float], bin_count: int) -> list[tuple[float,float]]:
    """Returns a list of the lower and upper cutoffs for each of the equal frequency bins"""
    # get a sorted list of the items to help identify where cuts should be made
    sorted_items: list[int|float] = list(sorted(items))
    # use a cursor to track the index of the last cut made
    last_cut: int = 0
    # use a variables to track how many more bins and items are left
    bins_remaining: int = bin_count
    items_remaining: int = len(sorted_items)
    # create a variable to hold the identified boundaries
    boundaries: list[tuple[float,float]] = []
    # loop to find more cuts until finished
    while bins_remaining > 0:
        # determine how many items should be in this next bin
        items_in_bin: int = min(items_remaining, int(round(items_remaining/bins_remaining)))
        # determine the index where the next cut should be made to include that many items
        next_cut = last_cut + items_in_bin
        # get the values at the relevant indices to make bin cuts
        bin_min = sorted_items[max(0,last_cut)]
        bin_max = sorted_items[min(next_cut, len(sorted_items)-1)]
        # add these values to the boundaries to be returned
        boundaries.append((bin_min, bin_max))
        # decrement bins and items remaining
        bins_remaining -= 1
        items_remaining -= items_in_bin
        # mark this cut as the last cut for the next iteration
        last_cut = next_cut
    return boundaries


# def main():
#     # z_score_norm([5, 4, 6, 7, 5.5])
#     # merge_uncommon(["Bill", "Bill", "James", "Stan", "Stan", "Stan"],
#     #                max_categories=2)


# main()