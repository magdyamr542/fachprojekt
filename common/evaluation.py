import numpy as np
from common.features import IdentityFeatureTransform
from collections import defaultdict, OrderedDict
from common.task import (
    load_ground_truths, 
    get_best_bag_of_features_histograms,
    intersection_over_union
)
from common.visualization import plot_results

class CrossValidation(object):

    def __init__(self, category_bow_dict, n_folds):
        """Initialisiert die Kreuzvalidierung ueber gegebnen Daten

        Params:
            category_bow_dict: Dictionary, das fuer jede Klasse ein ndarray mit Merkmalsvektoren
                (zeilenweise) enthaelt.
            n_folds: Anzahl von Ausschnitten ueber die die Kreuzvalidierung berechnet werden soll.

        """
        self.__category_bow_list = list(category_bow_dict.items())
        self.__n_folds = n_folds

    def validate(self, classifier, feature_transform=None):
        """Berechnet eine Kreuzvalidierung ueber die Daten,

        Params:
            classifier: Objekt, das die Funktionen estimate und classify implementieren muss.
            feature_transform: Objekt, das die Funktionen estimate und transform implementieren
                muss. Optional: Falls None, wird keine Transformation durchgefuehrt.

        Returns:
            tuple: (crossval_overall_result, crossval_class_results)
            
            crossval_overall_result: Fehlerrate in Prozent über allen Daten
            crossval_class_results: Klassenspezifische Fehlerraten in Prozent
        """
        if feature_transform is None:
            feature_transform = IdentityFeatureTransform()

        crossval_overall_list = []
        crossval_class_dict = defaultdict(list)

        for fold_index in range(self.__n_folds):
            train_bow, train_labels, test_bow, test_labels = self.corpus_fold(fold_index)
            feature_transform.estimate(train_bow, train_labels)

            train_feat = feature_transform.transform(train_bow)
            test_feat = feature_transform.transform(test_bow)
            classifier.estimate(train_feat, train_labels)
            estimated_test_labels = classifier.classify(test_feat)

            classifier_eval = ClassificationEvaluator(estimated_test_labels, test_labels)
            crossval_overall_list.append(list(classifier_eval.error_rate()))
            crossval_class_list = classifier_eval.category_error_rates()
            for category, err, n_wrong, n_samples in crossval_class_list:
                crossval_class_dict[category].append([err, n_wrong, n_samples])

        crossval_overall_mat = np.array(crossval_overall_list)
        crossval_overall_result = CrossValidation.__crossval_results(crossval_overall_mat)

        crossval_class_results = []
        for category in sorted(crossval_class_dict.keys()):
            crossval_class_mat = np.array(crossval_class_dict[category])
            crossval_class_result = CrossValidation.__crossval_results(crossval_class_mat)
            crossval_class_results.append((category, crossval_class_result))

        return crossval_overall_result, crossval_class_results

    @staticmethod
    def __crossval_results(crossval_mat):
        # Relative number of samples
        crossval_weights = crossval_mat[:, 2] / crossval_mat[:, 2].sum()
        # Weighted sum over recognition rates for all folds
        crossval_result = (crossval_mat[:, 0] * crossval_weights).sum()
        return crossval_result

    def corpus_fold(self, fold_index: int):
        """Berechnet eine Aufteilung der Daten in Training und Test

        Params:
            fold_index: Index des Ausschnitts der als Testdatensatz verwendet werden soll.

        Returns:
            Splits the data at index `fold_index` with the first half being the training data set and the second half is the testing data set.
            (training_bow_mat, training_label_mat, test_bow_mat, test_label_mat)
        """
        training_bow_mat = []
        training_label_mat = []
        test_bow_mat = []
        test_label_mat = []

        for category, bow_mat in self.__category_bow_list:
            n_category_samples = bow_mat.shape[0]
            #
            # Erklaeren Sie nach welchem Schema die Aufteilung der Daten erfolgt.
            #
            # Die Daten werden in self.__n_folds-Schritten pro Kategorie durchlaufen und die Indizes zu den Test-Indizes hinzugefügt.
            # Die restlichen Indizes werden zu den Training-Indizes hinzugefügt.

            # Select indices for fold_index-th test fold, remaining indices are used for training
            test_indices = list(range(fold_index, n_category_samples, self.__n_folds))
            train_indices = [train_index for train_index in range(n_category_samples)
                             if train_index not in test_indices]
            category_train_bow = bow_mat[train_indices, :]
            category_test_bow = bow_mat[test_indices, :]
            # Construct label matrices ([x]*3 --> [x, x, x])
            category_train_labels = np.array([[category] * len(train_indices)])
            category_test_labels = np.array([[category] * len(test_indices)])

            training_bow_mat.append(category_train_bow)
            training_label_mat.append(category_train_labels.T)
            test_bow_mat.append(category_test_bow)
            test_label_mat.append(category_test_labels.T)

        training_bow_mat = np.vstack(tuple(training_bow_mat))
        training_label_mat = np.vstack(tuple(training_label_mat))
        test_bow_mat = np.vstack(tuple(test_bow_mat))
        test_label_mat = np.vstack(tuple(test_label_mat))

        return training_bow_mat, training_label_mat, test_bow_mat, test_label_mat

class ClassificationEvaluator(object):

    def __init__(self, estimated_labels, groundtruth_labels):
        """Initialisiert den Evaluator fuer ein Klassifikationsergebnis
        auf Testdaten.

        Params:
            estimated_labels: ndarray (N x 1) mit durch den Klassifikator
                bestimmten Labels.
            groundtruth_labels: ndarray (N x 1) mit den tatsaechlichen Labels.

        """
        self.__estimated_labels = estimated_labels
        self.__groundtruth_labels = groundtruth_labels
        #
        # Bestimmen Sie hier die Uebereinstimmungen und Abweichungen der
        # durch den Klassifikator bestimmten Labels und der tatsaechlichen
        # Labels
        n_wrong = np.count_nonzero(self.__groundtruth_labels != self.__estimated_labels)
        self.__error_rates = (n_wrong / len(self.__groundtruth_labels)) * 100, n_wrong, len(self.__groundtruth_labels)


    def error_rate(self, mask=None):
        """Bestimmt die Fehlerrate auf den Testdaten.

        Params:
            mask: Optionale boolsche Maske, mit der eine Untermenge der Testdaten
                ausgewertet werden kann. Nuetzlich fuer klassenspezifische Fehlerraten.
                Bei mask=None werden alle Testdaten ausgewertet.
        Returns:
            tuple: (error_rate, n_wrong, n_samlpes)
            error_rate: Fehlerrate in Prozent
            n_wrong: Anzahl falsch klassifizierter Testbeispiele
            n_samples: Gesamtzahl von Testbeispielen
        """
        if mask is None:
            return self.__error_rates
        else:
            estimated = self.__estimated_labels[mask]
            groundtruth = self.__groundtruth_labels[mask]
            n_wrong = np.count_nonzero(groundtruth != estimated)
            return (n_wrong / len(groundtruth)) * 100, n_wrong, len(groundtruth)


    def category_error_rates(self):
        """Berechnet klassenspezifische Fehlerraten

        Returns:
            list von tuple: [ (category, error_rate, n_wrong, n_samlpes), ...]
            category: Label der Kategorie / Klasse
            error_rate: Fehlerrate in Prozent
            n_wrong: Anzahl falsch klassifizierter Testbeispiele
            n_samples: Gesamtzahl von Testbeispielen
        """
        unique_categories = sorted(OrderedDict.fromkeys([label[0] for label in self.__groundtruth_labels]))

        results = []
        for category in unique_categories:
            subset_mask = np.apply_along_axis(lambda x: x[0] == category, axis=1, arr=self.__groundtruth_labels)
            results.append((category, *self.error_rate(mask=subset_mask)))

        return results

class SegmentfreeWordSpottingEvaluator(object):

    def __init__(self, 
        img_path: str, 
        n_centroids: int,
        step_size: int,
        rel_threshold=0.5
    ) -> None:
        """
        Params:
            img_path: Path to image
            n_centroids: Number of clusters for features
            step_size: Size of steps for sift descriptors
            rel_treshold: Threshold for relevant findings
        """
        if rel_threshold < 0 or rel_threshold > 1:
            raise Exception(f"rel_threshold needs to be in the interval [0, 1], is {rel_threshold}")

        self.__img_path = img_path
        self.__n_centroids = n_centroids
        self.__step_size = step_size
        self.__visual_words = load_ground_truths(self.__img_path.rsplit('.')[0])
        self.__rel_threshold = rel_threshold

    def validate(self, max_eval_length=10) -> tuple[float, float, float]:
        # average_precision, average_recall, all_precisions, all_recalls
        """Validates all words in the given document.

        Params:
            max_eval_length: Maximum length for evaluation list to check if results are relevant

        Returns:
            (average_precision, 
             average_recall,
             avg_mean_prec,
             overall_precision, 
             overall_recall,
             overall_mean_prec)
        """
        overall_precision: list[float] = []
        overall_recall: list[float] = []
        overall_mean_prec: list[float] = []

        print(f"Evaluating {self.__img_path} with {len(self.__visual_words)} words")

        for idx in range(len(self.__visual_words))[:1]:
            print(f"Validating image '{self.__img_path}' with word {idx}")
            simple_precision, simple_recall, mean_prec = self.crossvalidate(idx, max_eval_length)
            overall_precision.append(simple_precision)
            overall_recall.append(simple_recall)
            overall_mean_prec.append(mean_prec)

        avg_prec = sum(overall_precision) / len(overall_precision)
        avg_rec = sum(overall_recall) / len(overall_recall)
        avg_mean_prec = sum(overall_mean_prec) / len(overall_mean_prec)

        return avg_prec, avg_rec, avg_mean_prec, overall_precision, overall_recall, overall_mean_prec 

    def crossvalidate(self, word_index: int, max_eval_length=10) -> tuple[float, float, float]:
        """Validates the word at index `word_index` in the given document.

        Params:
            word_index: Index of word in visual words
            max_eval_length: Maximum length of the list to evaluate precision and recall. 
                If 0, then the #occurences of the word in the document is used.

        Returns:
            (precision, recall, mean_precision) for word at `word_index` in `self.__visual_words`
        """
        if word_index < 0 or word_index >= len(self.__visual_words):
            raise Exception("Invalid word index")

        word = self.__visual_words[word_index]
        print(f"Evaluating word {word}")

        if max_eval_length < 1:
            print("Set max_eval_length to 10 because of invalid value given")
            max_eval_length = 10

        # evaluate with get_best_bag_of_features_histograms
        result = get_best_bag_of_features_histograms(
            self.__img_path, word[:4],
            self.__n_centroids,
            self.__step_size)
    
        # evalutation
        rel_words = list(filter(lambda x: x[-1] == word[-1], self.__visual_words))        
        rel_results = result[:max_eval_length]
        is_rel = self.__relevant_results(rel_words, rel_results)

        simple_precision = sum(is_rel) / len(rel_results)
        simple_recall = sum(is_rel) / len(rel_words)
        mean_prec, _ = SegmentfreeWordSpottingEvaluator.__mean_precision(is_rel, len(rel_words))

        return simple_precision, simple_recall, mean_prec

    @staticmethod
    def __mean_precision(relevant, total_relevant_words: int) -> tuple[float, list]:
        avg_precision = []

        for k in range(len(relevant)):
            if relevant[k] == 1:
                pc = sum(relevant[:k + 1])
                avg_precision.append(pc / (k + 1))

        return sum(avg_precision) / total_relevant_words, avg_precision
    
    def __relevant_results(self, gt_list, result_list):
        elem_relevant = []

        for result in result_list:
            coords = result['window']
            relevant = 0

            for gt in gt_list:
                if intersection_over_union(coords, gt[:4]) >= self.__rel_threshold:
                    relevant = 1
                    break
            
            elem_relevant.append(relevant)
        return elem_relevant
