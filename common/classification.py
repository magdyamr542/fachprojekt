import numpy as np
from scipy.spatial.distance import cdist
from common.features import BagOfWords, IdentityFeatureTransform
from operator import itemgetter
import operator



class KNNClassifier(object):

    def __init__(self, k_neighbors, metric):
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
        # calculate the distances
        if self.__train_samples is None or self.__train_labels is None:
                    raise ValueError('Classifier has not been "estimated", yet!')
        else:
            tst_labels = []
            distances = cdist(test_samples, self.__train_samples, self.__metric)
            sorted_indicies = np.argsort(distances,axis=1)
            k_nearest_neighbours = sorted_indicies[:,:self.__k_neighbors]
            for index_list in k_nearest_neighbours:
                k_nearest_labels = (self.__train_labels[index_list])
                label = BagOfWords.most_freq_words(k_nearest_labels.flatten(),1)
                tst_labels.append(label)
                
            return np.array(tst_labels)





