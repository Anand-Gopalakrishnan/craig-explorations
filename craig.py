##############################################################################################
##############################################################################################
##############################################################################################

import numpy as np
import heapq
import time
from sklearn.metrics.pairwise import pairwise_distances

##############################################################################################
##############################################################################################
##############################################################################################

class FacilityLocation:
    def __init__(self, similarity_matrix: np.ndarray, column_indices: list[int], alpha: float = 1.0):
        '''
        Parameters:
        - similarity_matrix: np.ndarray, shape [N, N]
        - column_indices: list[int], indices of columns of similarity_matrix
        - alpha: float, a scaling factor
        '''
        self.similarity_matrix = similarity_matrix
        self.current_value = 0
        self.current_maximum = np.zeros(similarity_matrix.shape[0])
        self.gains = []
        self.alpha = alpha # a constant?
        self.f_norm_value = self.alpha / self.calculate_f_norm(column_indices)
        self.norm = 1.0 / self.increment(column_indices, []) # inputting a false-like value to trigger the default

    def calculate_f_norm(self, column_subset):
        return self.similarity_matrix[:, column_subset].max(axis=1).sum()

    def max_selector(self, subset, index):
        if len(subset + [index]) > 1:
            selected_value = np.maximum(self.current_maximum, self.similarity_matrix[:, index])
        else:
            selected_value = self.similarity_matrix[:, index]
        return selected_value

    def increment(self, subset, index):
        if not index: # normalization (providing a reference point)
            return np.log(1 + self.alpha * 1)
        else:
            selected_value = self.max_selector(subset, index)
            return self.norm * np.log(1 + self.f_norm_value * selected_value.sum()) - self.current_value

    def add(self, subset, index):
        old_current_value = self.current_value
        self.current_maximum = self.max_selector(subset, index)
        self.current_value = self.norm * np.log(1 + self.f_norm_value * self.current_maximum.sum())
        self.gains.extend([self.current_value - old_current_value])
        return self.current_value


##############################################################################################
##############################################################################################
##############################################################################################


def lazy_greedy_heap(facility_location: FacilityLocation, column_indices: list[int], num_points: int):
    '''
    Constructs the sequence of indices in the subset, ordered by facility location.

    Parameters:
    - facility_location: FacilityLocation
    - column_indices: list[int], indices of columns of similarity_matrix in facility_location,
    - num_points: int, number of points to choose for the subset

    Returns:
    - subset: list, sequence of indices in the subset
    - values: list, the facility locations as the subset is constructed
    '''
    current_value = 0
    subset = []
    values = []

    # Evaluate facility locations on an empty subset of columns [] to construct the max-heap `order`.
    order = []
    heapq.heapify_max(order)
    for index in column_indices:
        heapq.heappush_max(order, (facility_location.increment(subset, index), index))

    # While the heap `order` is not empty and we still have points to select...
    while order and len(subset) < num_points:

        # Pop the largest element in the heap `order` and get its facility location
        element = heapq.heappop_max(order)
        improvement = facility_location.increment(subset, element[1])

        # If the improvement is not insignificant...
        if improvement >= 0:

            # If `order` is now empty, add the last popped facility location (and column) to our subset.
            # The while-condition will now break on the next loop, so we are done.
            if not order:
                current_value = facility_location.add(subset, element[1])
                subset.append(element[1])
                values.append(current_value)

            # Otherwise...
            else:
                # Pop the largest element in the heap `order`
                largest = heapq.heappop_max(order)

                # If the `improvement` is larger than the facility location of the `largest`,
                # add it to our subset (shortens heap length by 1). 
                if improvement >= largest[0]:
                    current_value = facility_location.add(subset, element[1])
                    subset.append(element[1])
                    values.append(current_value)
                # Otherwise, push the `improvement` into the heap `order` (resets heap length).
                else:
                    heapq.heappush_max(order, (improvement, element[1]))

                # Push the largest element back into the heap `order` (resets heap length).
                heapq.heappush_max(order, largest)

    return subset, values


##############################################################################################
##############################################################################################
##############################################################################################


def similarity(feature_array: np.ndarray, metric: str):
    '''
    Computes the similarity between each pair of examples in feature_array.

    Parameters
    - feature_array: np.ndarray, shape [N, d]
    - metric: str, one of ['cosine', 'euclidean', 'l1']

    Returns
    - similarity_matrix: np.ndarray, shape [N, N]
    - time_elapsed: float, time taken to compute the similarities
    '''
    # Calculate pair-wise distances between all examples in `feature_array` and the time taken for this
    time_start = time.time()
    distances = pairwise_distances(feature_array, metric=metric, n_jobs=1)
    time_elapsed = time.time() - time_start

    # Compute the similarity matrix for the examples in `feature_array`, depending on the given metric
    if metric == 'cosine':
        similarity_matrix = 1 - distances
    elif metric == 'euclidean' or metric == 'l1':
        similarity_matrix = np.max(distances) - distances
    else:
        raise ValueError(f'unknown metric: {metric}')

    return similarity_matrix, time_elapsed


##############################################################################################
##############################################################################################
##############################################################################################


def get_facility_location_submodular_order(similarity_matrix: np.ndarray, num_points: int, 
                                           weights: np.ndarray | None = None):
    '''
    Generate the order of points selected by facility location, and compute sizes of 
    clusters associated with each selected point.

    Parameters
    - similarity_matrix: np.ndarray, shape [N, N]
    - num_points: int, number of points to select
    - weights: np.ndarray (optional), optinally weight the cluster sizes

    Returns
    - order: np.ndarray, shape [B], order of points selected by facility location
    - size: np.ndarray, shape [B], sizes of clusters associated with each selected point
    - greedy_time: float, time taken to compute the ordered sequence of indices

    Note: B = N * (subset-size)
    '''
    
    # Create a facility location object and generate an order of points, recording the time taken
    column_indices = list(range(similarity_matrix.shape[0]))
    time_start = time.time()
    facility_location = FacilityLocation(similarity_matrix, column_indices)
    order, _ = lazy_greedy_heap(facility_location, column_indices, num_points)
    greedy_time = time.time() - time_start

    order = np.asarray(order, dtype=np.int64)
    size = np.zeros(num_points, dtype=np.float64)

    # Construct the sizes of clusters associated with each point;
    # if `weights` are provided, weight the distribution appropriately
    for i in range(similarity_matrix.shape[0]):
        if weights is None:
            size[np.argmax(similarity_matrix[i, order])] += 1
        else:
            size[np.argmax(similarity_matrix[i, order])] += weights[i]

    return order, size, greedy_time


##############################################################################################
##############################################################################################
##############################################################################################


def facility_location_order(class_: int, feature_array: np.ndarray, feature_labels: np.ndarray, 
                            metric: str, num_per_class, weights: np.ndarray | None = None):
    '''
    Computes facility location order and cluster sizes for one particular output class

    Parameters
    - class_: int, an output class of the network
    - feature_array: np.ndarray, the full set of training data features
    - feature_labels: np.ndarry, the full set of training data labels,
    - metric: str, one of ['cosine', 'euclidean', 'l1']
    - num_per_class: int, number of points to pick for this output class (class_)
    - weights: np.ndarray (optional), optionally weight the cluster sizes

    Returns
    - class_indices[order]: np.ndarray, ordered sequence of indices for this output class (class_)
    - cluster_size: np.ndarray, sizes of clusters associated with each selected point
    - greedy_time: float, time taken to compute the ordered sequence of indices
    - similarity_matrix_time: float, time taken to compute the similarities
    '''

    # Array of training example indicies labelled with class `class_`
    class_indices = np.where(feature_labels == class_)[0]

    print(f'Computing facility location order for: {class_}')

    # Compute the similarity matrix for the given array of features
    similarity_matrix, similarity_matrix_time = similarity(feature_array[class_indices], metric=metric)

    # Compute the facility location order and cluster-size based on the given similarity matrix
    order, cluster_size, greedy_time = get_facility_location_submodular_order(
        similarity_matrix, num_per_class, weights)
    
    return class_indices[order], cluster_size, greedy_time, similarity_matrix_time


##############################################################################################
##############################################################################################
##############################################################################################


def preserve_sum_and_proportions(array: np.ndarray, required_sum: int):
    '''
    Helper function that scales a required array of intergers while preserving the
    proportions and sums as well as possible.

    Parameters:
    - array: np.ndarray, array to be scaled
    - required_sum: int, the required total sum after scaling

    Returns:
    - new_array: the final scaled array
    '''
    # Scale the array appropriately
    scaled_array = (array / np.sum(array)) * required_sum

    # Attempt to make a new array of intergers through rounding
    new_array = np.int32(np.round(scaled_array))
    difference = np.int32(np.sum(new_array - scaled_array))

    # Correct mistakes to the sum that occur due to the above step (ad hoc correction)
    if difference < 0: # new sum is too large
        new_array[:np.abs(difference)] += 1
    elif difference > 0: # new sum is too small
        new_array[:np.abs(difference)] -= 1

    return new_array


##############################################################################################
##############################################################################################
##############################################################################################


def get_orders_and_weights(feature_array: np.ndarray, num_points: int, metric: str,  
                           feature_labels: np.ndarray | None = None, weights: np.ndarray | None = None, 
                           equal_num: bool = False):
    '''
    Construct the final ordered sequence of indices in the subset and their associated weights.

    Parameters
    - feature_array: np.ndarray, shape [N, d]
    - num_points: int, number of points to select
    - metric: str, one of ['cosine', 'euclidean', 'l1'], for similarity
    - feature_labels: np.ndarray, shape [N], integer class labels for C classes

    Returns
    - order_mg: np.ndarray, shape [num_points], type int64,
                order points by their marginal gain in FL objective (largest gain first)
    - weights_mg: np.ndarray, shape [num_points], type float32, sums to 1
    '''

    # If there are no feature labels, assign every point to the same class
    if feature_labels is None:
        feature_labels = np.zeros(feature_array.shape[0], dtype=np.int32)

    # The classes are the collection of unique labels; let C be the number of classes
    classes = np.unique(feature_labels)
    C = len(classes)

    # Force the class-sizes to be as equally distributed as possible (if that parameter is set)
    if equal_num:

        # Compare the amount of elements in each class vs. a distribution with all class-sizes equal
        class_nums = np.asarray([np.sum(feature_labels == class_) for class_ in classes])
        num_per_class = int(np.ceil(num_points / C)) * np.ones(C, dtype=np.int32)
        minority = class_nums < np.ceil(num_points / C)

        # If there are *extra* elements, divide these up as equally as possible 
        if np.sum(minority) > 0:
            extra = sum([max(0, np.ceil(num_points / C) - class_nums[class_]) for class_ in classes])
            for class_ in classes[~minority]:
                num_per_class[class_] += int(np.ceil(extra / sum(minority)))

    # Otherwise, keep the proportions of class-sizes as they are (good as default)
    else:
        class_nums = np.asarray([np.sum(feature_labels == class_) for class_ in classes])
        num_per_class = preserve_sum_and_proportions(class_nums, num_points)

    # Get orders and cluster sizes (and computation times) for each class among the feature labels
    order_mg_all, cluster_sizes_all, greedy_times, similarity_times = zip(*map(
        lambda class_: facility_location_order(class_, feature_array, feature_labels, 
                                               metric, num_per_class[class_], weights), classes))

    order_mg, weights_mg = [], []
    if equal_num:
        proportions = np.round([len(order_mg_all[i]) for i in range(len(order_mg_all))])
    else:
        # Compute fraction of data in each class, followed by relative sizes of classes
        # with respect to the smallest class
        class_ratios = np.divide([np.sum(feature_labels == i) for i in classes], feature_array.shape[0])
        proportions = np.round(class_ratios / np.min(class_ratios))
        print(f'Selecting with ratios {np.array(class_ratios)}')
        print(f'Class proportions {np.array(proportions)}')

    # Constructing the full set final indices and weights
    for i in range(int(np.round(np.max([len(order_mg_all[c]) / proportions[c] for c in classes])))):
        for c in classes:
            index = slice(i * int(proportions[c]), int(min(len(order_mg_all[c]), (i + 1) * proportions[c])))
            order_mg = np.append(order_mg, order_mg_all[c][index])
            weights_mg = np.append(weights_mg, cluster_sizes_all[c][index])

    # Type-casting
    order_mg = np.array(order_mg, dtype=np.int32)
    weights_mg = np.array(weights_mg, dtype=np.float32)

    ordering_time = np.max(greedy_times)
    similarity_time = np.max(similarity_times)

    return order_mg, weights_mg, ordering_time, similarity_time

