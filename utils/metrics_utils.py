"""
FILE TO CONTAIN VARIOUS METHODS OF CALCULATING PERFORMANCE METRICS FOR DIFFERENT MODELS
"""

import os
import shutil

import numpy      as np
import tensorflow as tf
import sklearn import svm


class Metrics():
    def __init__(self, output_dims, verbose=True):
        """
        Args:
            :output_dims: Output dimensions of the model, used to verify the shape of predictions
            :verbose:     Setting verbose command
        """
        self.output_dims=output_dims
        self.verbose=verbose
        self.correct_predictions=0
        self.total_predictions=0
        self.predicitons_array=[]

    def _save_prediction(label, prediction, name):
        if not os.path.isdir('temp'):
            os.mkdir('temp')

        np.save(name, (prediction, label))


    def log_prediction(label, predictions, names, method):
        """
        Args:
            :label:            Ground truth label of the video(s) used to generate the predictions
            :predictions:      The output predictions from the model accross all batches
            :name:             The name(s) of the video(s) currently being classified
            :method:           The classification method to use
        Return:
            :current_accuracy: The current classification accuracy of all videos
                               passed through this object accross multiple calls of this method
        """
        if method == 'default':
            current_accuracy = _default_classify(label, predictions, names)

        elif method == 'svm':
            current_accuracy = _svm_log(label, predictions, names)

        else:
            print "Error: Invalid classification method"
            exit()

        return current_accuracy


    def total_classification(method):
        """
        Args:
            :label:            Ground truth label of the video(s) used to generate the predictions
            :predictions:      The output predictions from the model accross all batches
            :name:             The name(s) of the video(s) currently being classified
            :method:           The classification method to use
        Return:
            :current_accuracy: The current classification accuracy of all videos
                               passed through this object accross multiple calls of this method
        """
        if method == 'default':
            current_accuracy = get_accuracy()

        elif method == 'svm':
            current_accuracy = _svm_classify()

        else:
            print "Error: Invalid classification method"
            exit()

        return current_accuracy


    def _default_classify(label, predictions, names):
        """
        Default argmax classification
        Args:
            :label:            Ground truth label of the video(s) used to generate the predictions
            :predictions:      The output predictions from the model accross all batches
            :name:             The name(s) of the video(s) currently being classified

        Return:
            :current_accuracy: The current classification accuracy of all videos
                               passed through this object accross multiple calls of this method
        """
        # Average predictions if single video is broken into clips
        if len(predictions.shape)!=2:
            predictions = np.mean(predictions, 1)

        # END IF

        prediction = np.mean(predictions, 0).argmax()

        if

        if self.verbose:
            print "vidName: ",names
            print "label:  ", label
            print "prediction: ", prediction

        self.predictions_array.append((prediction, label))
        self.total_predictions += 1
        if int(prediction) == int(label):
            self.correct_predictions += 1

        current_accuracy = self.correct_predictions / float(self.total_predictions)
        return current_accuracy


    def _svm_log(label, predictions, names):
        """
        Stores predicitons until testing is completed and an svm is trained
        Args:
            :label:            Ground truth label of the video(s) used to generate the predictions
            :predictions:      The output predictions from the model accross all batches
            :name:             The name(s) of the video(s) currently being classified

        Return:
            :current_accuracy: Accuracy is not calculated until all testing videos have been logged
                               and the svm can be trained
        """
        _save_prediction(label, predictions, names)

        return -1


    def _svm_classify():
        """
        Final classification of predictions saved to temp folder using a linear svm
        Args:
            None
        Return:
            :current_accuracy: The current classification accuracy of all videos
                               passed through this object accross multiple calls of this method
        """

        model_output = []
        labels = []
        names = []


        for f in os.listdir('temp'):
            if f in names:
                ind = names.index(f)
                data = np.load(f)
                model_output[ind].append(data[0])
            else:
                data = np.load(f)
                model_output.append([data[0]])
                names.append(f)
                labels.append(data[1])
        averaged_output = []
        for output in range(len(model_output)):
            names.append(f)
            data = np.load(f)
            model_output.append(data[0])
            labels.append(data[1])


        classifier = svm.SVC(kernel='linear')

        classifier.fit(model_output, labels)

        predictions = classifier.predict(model_output)

        if len(predictions.shape)!=2:
            predictions = np.mean(predictions, 1)

        # END IF

        prediction = np.mean(predictions, 0).argmax()

        if self.verbose:
            print "vidName: ",names
            print "label:  ", label
            print "prediction: ", prediction

        self.predictions_array.append((prediction, label))
        self.total_predictions += 1
        if int(prediction) == int(label):
            self.correct_predictions += 1

        shutil.rmtree('temp')

        current_accuracy = self.correct_predictions / float(self.total_predictions)
        return current_accuracy


    def get_predictions_array():
        """
        Args:
            None
        Return:
            :predictions_array:  Array of predictions with each index containing (prediction, ground_truth_label)
        """
        return self.predictions_array

    def get_accuracy():
        """
        Args:
            None
        Return:
            Total accuracy of classifications
        """
        return self.correct_predictions / float(self.total_predictions)

    def clear_all():
        """
        Clear all parameters (correct_predictions, total_predictions, predicitons_array)
        Args:
            None
        Return:
            None
        """
        self.correct_predictions = 0
        self.total_predictions = 0
        self.predictions_array = []
