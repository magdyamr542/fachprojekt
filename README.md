# Fachprojekt 'Dokumentenanalyse'

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


- nms times:
    - iou without numpy: 28.66596299999992
    - iou with numpy: 113.08130100000017


