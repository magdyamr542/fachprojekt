# Fachprojekt 'Dokumentenanalyse'

## Results

- 2700270
    - and (3)
        - n_centroids: 100 - step_size: 30 - max_eval_length: 10
            - prec=0.6 rec=0.857 mean_prec=0.857
        - n_centroids: 150 - step_size: 30 - max_eval_length: 10
            - prec=0.6 rec=0.857 mean_prec=0.857
        - n_centroids: 200 - step_size: 30 - max_eval_length: 10
            - prec=0.6 rec=0.857 mean_prec=0.857
        - n_centroids: 200 - step_size: 20 - max_eval_length: 10
            - prec=0.8 rec=1.143 mean_prec=1.109
        - n_centroids: 200 - step_size: 10 - max_eval_length: 10
            - prec=0.9 rec=1.286 mean_prec=1.286
    - instructions (4)
        - n_centroids: 100 - step_size: 30 - max_eval_length: 10
            - prec=0.7 rec=3.5 mean_prec=3.279
        - n_centroids: 150 - step_size: 30 - max_eval_length: 10
            - prec=0.8 rec=4.0 mean_prec=3.838
        - n_centroids: 200 - step_size: 30 - max_eval_length: 10
            - prec=0.7 rec=3.5 mean_prec=3.366
        - n_centroids: 200 - step_size: 20 - max_eval_length: 10
            - prec=1.0 rec=5.0 mean_prec=5.0
        - n_centroids: 200 - step_size: 10 - max_eval_length: 10
            - prec=1.0 rec=5.0 mean_prec=5.0
    - october (5)
        - n_centroids: 100 - step_size: 30 - max_eval_length: 10
            - prec=0.6 rec=2.0 mean_prec=2.0
        - n_centroids: 150 - step_size: 30 - max_eval_length: 10
            - prec=0.5 rec=1.667 mean_prec=1.667
        - n_centroids: 200 - step_size: 30 - max_eval_length: 10
            - prec=0.6 rec=2.0 mean_prec=2.0
        - n_centroids: 200 - step_size: 20 - max_eval_length: 10
            - prec=0.8 rec=2.667 mean_prec=2.667
        - n_centroids: 200 - step_size: 10 - max_eval_length: 10
            - prec=1.0 rec=3.333 mean_prec=3.333
    - officers (83)
        - n_centroids: 100 - step_size: 30 - max_eval_length: 10
            - prec=0.5 rec=5.0 mean_prec=5.0
        - n_centroids: 150 - step_size: 30 - max_eval_length: 10
            - prec=0.5 rec=5.0 mean_prec=4.834
        - n_centroids: 200 - step_size: 30 - max_eval_length: 10
            - prec=0.5 rec=5.0 mean_prec=5.0

## Probleme

- Kurze Wörter finden

## Evaluation

- word: and
    - max_eval_len: 20
        - old technique
            - prec=0.15
            - rec=0.42857142857142855 
            - mean_prec=0.3095238095238095
        - new technique (smaller x steps for sliding window, step_size / 2)
            - prec=0.35
            - rec=1.0
            - mean_prec=0.9404761904761905
    - max_eval_len: 10
        - old technique
            - prec=0.3
            - rec=0.42857142857142855
            - mean_prec=0.39285714285714285
        - new technique (smaller x steps for sliding window, step_size / 2)
            - prec=0.7
            - rec=1.0
            - mean_prec=0.9821428571428571

- get_best_bag_of_features_histograms optimization
    - not optimized
        - outer time: 39.97936000000004
        - acc inner time: 39.839147000010826
        - outer time: 40.102468000000044
        - acc inner time: 39.9182350000101
    - optimized
        - outer time: 3.1077660000000833
        - acc inner time: 2.9950079999857735
        - outer time: 2.808492000000115
        - acc inner time: 2.6949609999501263

- nms times:
    - iou without numpy: 28.66596299999992
    - iou with numpy: 113.08130100000017
    - using only for-loops instead of while and for-loop
        - no typing of window
            - time nms: 53.92093700000032
        - initial window typing
            - time nms: 29.794601000000057
        - typing all windows
            - time nms: 30.71906799999988
            - time nms: 30.856462000000192
        - started use only for-loops
            - time nms: 23.379664000000048
            - time nms: 21.630252000000382


