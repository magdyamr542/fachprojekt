import numpy as np
from scipy.spatial.distance import cdist
from common.features import BagOfWords


class KNNClassifier(object):

    def __init__(self, k_neighbors: int, metric):
        """Initialisiert den Klassifikator mit Meta-Parametern
        
        Params:
            k_neighbors: Anzahl der zu betrachtenden naechsten Nachbarn (int)
            metric: Zu verwendendes Distanzmass (string), siehe auch scipy Funktion cdist 
        """
        self.__k_neighbors = k_neighbors
        self.__metric = metric
        # Initialisierung der Membervariablen fuer Trainingsdaten als None. 
        self.__train_samples = None
        self.__train_labels = None

    def estimate(self, train_samples, train_labels):
        """Erstellt den k-Naechste-Nachbarn Klassfikator mittels Trainingdaten.
        
        Der Begriff des Trainings ist beim K-NN etwas irre fuehrend, da ja keine
        Modelparameter im eigentlichen Sinne statistisch geschaetzt werden. 
        Diskutieren Sie, was den K-NN stattdessen definiert.
        
        Hinweis: Die Funktion heisst estimate im Sinne von durchgaengigen Interfaces
                fuer das Duck-Typing
        
        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
            
            mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        self.__train_samples = train_samples
        self.__train_labels = train_labels

    def classify(self, test_samples):
        """Klassifiziert Test Daten.
        
        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            
        Returns:
            test_labels: ndarray, das Klassenlabels spaltenweise enthaelt (d x 1).
        
            mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        """
        if self.__train_samples is None or self.__train_labels is None:
            raise ValueError('Classifier has not been "estimated", yet!')

        result_labels = np.array([])
        for sample in test_samples:
            distances = cdist(self.__train_samples, [sample], self.__metric)

            if self.__k_neighbors == 1:
                label = self.__train_labels[np.argmin(distances)]
            else:
                label_dist = np.column_stack((self.__train_labels, distances))
                label_dist = label_dist[label_dist[:, 1].astype(np.float64).argsort()]

                k_nearest_labels = label_dist[0:self.__k_neighbors, 0]

                # Majority vote mit most_freq_words
                label = BagOfWords.most_freq_words(k_nearest_labels, n_words=1)

            if result_labels.shape[0] == 0:
                result_labels = np.array([label])
            else:
                result_labels = np.row_stack((result_labels, [label]))

        return np.asarray(result_labels)

            



