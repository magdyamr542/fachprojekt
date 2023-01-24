import numpy as np
import pickle
import PIL.Image as Image

from scipy.cluster.vq import kmeans2
from collections import deque
from os.path import abspath, join, exists as path_exists

from common.features import compute_sift_descriptors

import time

def get_best_bag_of_features_histograms(
    img_path: str, 
    coords: tuple[int, int, int, int],
    n_centroids: int,
    step_size=20,
    size=None,
    iters=20) -> list:
    """
    Params:
        img_path: Path to image file
        coords: (x1, y1, x2, y2) coordinates of the request image
        n_centroids: number of centroids used for the clustering
        step_size: the step size for computing sift descriptors of the main document. If 0, step_size = cell_size // 4
        size: Uses the value as cell size [Default = smaller request window dimension]
        iters: max iterations for clustering found features in main document
    """
    DEBUG = False
    INFO = True

    # load main document and crop the request document
    x1, y1, x2, y2 = coords
    document = Image.open(path_join('pages', '2700270.png'))
    doc_arr = np.asarray(document, dtype='uint8')
    req_arr = doc_arr[y1:y2, x1:x2]

    if size is None:
        cell_size = min(req_arr.shape)
    else:
        cell_size = size

    if step_size == 0:
        step_size = cell_size // 4

    subpath = img_path.rsplit('.', 1)[0]
    subpath = f"{subpath}_{cell_size}_sift-{step_size}_descriptors.p"
    pickle_path = path_join('pickle_data', 'sift_descriptors', subpath)

    if INFO:
        print(f"Using {cell_size=} and {step_size=}")

    # TODO: uncomment
    if path_exists(pickle_path) and False:
        if INFO:
            print(f"Loading pickled sift descriptors from file...")

        with open(pickle_path, 'rb') as fh:
            doc_frames, doc_desc = pickle.load(fh)
            doc_frames = np.asarray(doc_frames)
    else:
        # compute sift descriptors and pickle afterwards
        if INFO:
            print(f"Computing sift descriptors and pickling afterwards...")

        doc_frames, doc_desc = compute_sift_descriptors(doc_arr, cell_size=cell_size, step_size=step_size)
        doc_frames = np.asarray(doc_frames)

        with open(pickle_path, 'wb') as fh:
            to_dump = (doc_frames, doc_desc)
            pickle.dump(to_dump, fh)


    if INFO:
        print(f"Clustering... {len(doc_desc)} descriptors to {n_centroids} clusters")
    
    # cluster most significant features
    _, labels = kmeans2(doc_desc, n_centroids, iter=iters, minit='points')
    labels = np.array(labels, dtype='int')

    # get request image bag of features
    req_img_bof = get_bag_of_feature_labels_count(doc_frames, labels, n_centroids, x1, y1, x2, y2)

    # window height and width
    wheight = y2 - y1
    wwidth = x2 - x1

    # base coordinates for moving window
    wx = 0
    wy = 0
    wxx = wwidth
    wyy = wheight

    # store all results in a list like [{'window': (x1, y1, x2, y2), 'bof': bof}]
    w_bofs = []

    if INFO:
        print("starting sliding window computations")

    while wyy < doc_arr.shape[0]:
        time0 = time.process_time()
        bof = get_bag_of_feature_labels_count(doc_frames, labels, n_centroids, wx, wy, wxx, wyy)
        w_bofs.append({
            'window': (wx, wy, wxx, wyy),
            'bof': bof
        })
        time1 = time.process_time()
        print(f"Needed {time1 - time0}")

        wx += step_size
        wxx = wx + wwidth
        
        # check if row done
        if wxx > doc_arr.shape[1]:
            wx = 0
            wxx = wwidth
            wy = wy + step_size
            wyy = wy + wheight

            if DEBUG:
                print("Computing row...", wy)
    
    if INFO:
        print(f"Generated {len(w_bofs)} bag of feature counts")

    # compare all computed histograms to request image
    for subw in w_bofs:
        subw['diff'] = np.abs(subw['bof'] - req_img_bof).sum()

    if INFO:
        print("Applying non-maximum-suppression...")

    # use non-maximum-suppression to reduce overlapping frames
    final_bofs = non_maximum_suppresion(w_bofs)

    return final_bofs


def non_maximum_suppresion(all_bofs: list) -> list:
    """
    Params:
        - all_bofs - list of {'window': (x1, y1, x2, y2), 'bof': sift descriptor for window, 'diff': difference to request image descriptor}
    """
    result_bofs = []
    sub_results = deque()

    while len(all_bofs) > 0:
        mbof = all_bofs.pop(0)
        sub_results.append(mbof)
        w = mbof['window']

        for bof in all_bofs:
            if intersection_over_union(w, bof['window']) >= 0.5:
                sub_results.append(bof)
                all_bofs.remove(bof)
        
        # use best bof from sub_results and add to result_bofs
        sub_results = deque(sorted(sub_results, key=lambda x: x['diff']))

        if len(sub_results) > 0:
            result_bofs.append(sub_results[0])

        sub_results.clear()

    return sorted(result_bofs, key=lambda x: x['diff'])


def intersection_over_union(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = intersection_area / (box1_area + box2_area - intersection_area)

    return iou


def get_bag_of_feature_labels_count(frames: np.ndarray, labels: np.ndarray, n_clusters, x1, y1, x2, y2) -> np.ndarray:
    """Get for each descriptor its corresponding cluster label and count them."""
    count_arr = np.zeros(n_clusters, dtype='int')
    # TODO: binary search to min index of current window

    for idx, frame in enumerate(frames):
        if frame[1] < x1 or frame[1] > x2 or frame[0] < y1 or frame[0] > y2:
            continue
        else:
            count_arr[labels[idx]] += 1

    return count_arr


def path_join(*args):
    return abspath(join('..', 'data', 'gw', *args))


def load_ground_truths(ground_truth_file_name: str):
    """Params:
        - file name with ending '.gtp'
    """
    if not ground_truth_file_name.endswith('.gtp'):
        ground_truth_file_name = ground_truth_file_name + '.gtp'

    visual_words = []
    with open(path_join('ground_truth', ground_truth_file_name)) as f:
        for line in f.readlines():
            x1, y1, y2, x2, word = line.split()
            x1, y1, y2, x2 = int(x1), int(y1), int(x2), int(y2)
            visual_words.append((x1, y1, x2, y2, word))
    
    return visual_words
