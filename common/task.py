import numpy as np
import PIL.Image as Image

from scipy.cluster.vq import kmeans2
from collections import deque
from os.path import abspath, join

from common.features import compute_sift_descriptors

def get_best_bag_of_features_histograms(
        img_path: str, 
        coords: tuple[int, int, int, int],
        n_centroids: int,
        step_size=20,
        size=-1,
        iters=20
    ) -> list:
    """
    `Params`:
        img_path: Path to image file
        coords: (x1, y1, x2, y2) coordinates of the request image
        n_centroids: number of centroids used for the clustering
        step_size_x: the step size for computing sift descriptors of the main document.
        size: Uses the value as cell size [Default = smaller request window dimension]
        iters: max iterations for clustering found features in main document
    """
    INFO = False

    # load main document and crop the request document
    x1, y1, x2, y2 = coords
    document = Image.open(path_join('pages', img_path))
    doc_arr = np.asarray(document, dtype='uint8')
    req_arr = doc_arr[y1:y2, x1:x2]

    if size < 1:
        cell_size = min(req_arr.shape)
    else:
        cell_size = size

    if INFO:
        print(f"Using {cell_size=} and {step_size=}")

    doc_frames, doc_desc = compute_sift_descriptors(doc_arr, cell_size=5, step_size=20)
    doc_frames = np.asarray(doc_frames, dtype='int')

    if INFO:
        print(f"Clustering... {len(doc_desc)} descriptors to {n_centroids} clusters")
    
    # cluster features for discrete distinction
    _, labels = kmeans2(doc_desc, n_centroids, iter=iters, minit='points')
    labels = np.asarray(labels, dtype='int')

    # get request image bag of features
    req_img_bof = get_bag_of_feature_labels_count(doc_frames, labels, n_centroids, x1, y1, x2, y2)

    # window height and width
    wheight = y2 - y1
    wwidth = x2 - x1

    # use smaller x step size because language is written row-wise
    step_size_x = step_size // 2
    step_size_y = step_size

    # store all results in a list like [{'window': (x1, y1, x2, y2), 'bof': bof}]
    w_bofs = []

    if INFO:
        print("Starting sliding window computations")

    for row in range(0, doc_arr.shape[0], step_size_y):
        row_end = row + wheight

        bool_idx = doc_frames[:, 0] >= row
        row_subframes = doc_frames[bool_idx]
        row_sublabels = labels[bool_idx]

        bool_idx = row_subframes[:, 0] <= row_end
        row_subframes = row_subframes[bool_idx]
        row_sublabels = row_sublabels[bool_idx]

        for col in range(0, doc_arr.shape[1], step_size_x):
            col_end = col + wwidth
            bof = get_bag_of_feature_labels_count(row_subframes, row_sublabels, n_centroids, col, row, col_end, row_end)
            
            # compare computed histograms to request image
            w_bofs.append({
                'window': (col, row, col_end, row_end),
                'bof': bof,
                'diff': np.abs(bof - req_img_bof).sum()
            })

    if INFO:
        print(f"Generated {len(w_bofs)=} bag of feature counts")
        print("Applying non-maximum-suppression...")

    # use non-maximum-suppression to reduce overlapping frames
    #w_bofs = sorted(w_bofs, key=lambda x: x['diff'])
    final_bofs = non_maximum_suppresion(w_bofs[:5000])

    return final_bofs


def non_maximum_suppresion(all_bofs: list) -> list:
    """
    Params:
        all_bofs: list of {'window': np.array(x1, y1, x2, y2), 'bof': sift descriptor for window, 'diff': difference to request image descriptor}
    """
    result_bofs = []
    sub_results = deque()
    
    print(f"Starting with {len(all_bofs)=}")

    while all_bofs:
        mbof = all_bofs.pop(0)
        sub_results.append(mbof)
        win: tuple[int, int, int, int] = mbof['window']

        for bof in all_bofs:
            bof_win: tuple[int, int, int, int] = bof['window']

            if intersection_over_union(win, bof_win) >= 0.5:
                sub_results.append(bof)
                all_bofs.remove(bof)

        # use best bof from sub_results and add to result_bofs
        sub_results_sorted = sorted(sub_results, key=lambda x: x['diff'])

        if len(sub_results_sorted) > 0:
            result_bofs.append(sub_results_sorted[0])

        sub_results.clear()

    print(f"Ending with {len(result_bofs)=}")

    return sorted(result_bofs, key=lambda x: x['diff'])


def get_bag_of_feature_labels_count(
        frames: np.ndarray, 
        labels: np.ndarray, 
        n_clusters, 
        x1, y1, x2, y2
    ) -> np.ndarray:
    """Get for each descriptor its corresponding cluster label and count them."""
    count_arr = np.zeros(n_clusters, dtype='int')

    for idx, frame in enumerate(frames):
        if frame[0] > y2: 
            break
        elif frame[1] < x1 or frame[1] > x2 or frame[0] < y1:
            continue
        else:
            count_arr[labels[idx]] += 1
    
    return count_arr


def intersection_over_union(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """Params:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)
    Returns:
        Value between 0 and 1 depending on the overlap of box1 and box2."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = intersection_area / (box1_area + box2_area - intersection_area)

    return iou


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
