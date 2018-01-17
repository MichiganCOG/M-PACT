"""
FILE TO CONTAIN VARIOUS METHODS OF CALCULATING PERFORMANCE METRICS FOR DIFFERENT MODELS
"""

import os
import shutil

import numpy      as np
import tensorflow as tf
import sklearn    import svm


class Metrics():
    """
    A class containing methods to log and calculate classification metrics
    Methods:
        :__init__:
        :log_prediction:
        :total_classification:
        :get_accuracy:
        :get_predictions_array:
        :clear_all:
        :_save_prediction:
        :_default_classify:
        :_svm_classify:
    """

    def __init__(self, output_dims, logger, method, is_training, verbose=True):
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
        self.logger=logger
        self.method=method
        self.step=0
        self.is_training = is_training
        if self.is_training:
            self.log_name = 'train'
        else:
            self.log_name = 'test'

    def get_accuracy(self):
        """
        Args:
            None
        Return:
            Total accuracy of classifications
        """
        return self.correct_predictions / float(self.total_predictions)


    def get_predictions_array(self):
        """
        Args:
            None
        Return:
            :predictions_array:  Array of predictions with each index containing (prediction, ground_truth_label)
        """
        return self.predictions_array


    def clear_all(self):
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


    def log_prediction(self, label, predictions, names, step):
        """
        Args:
            :label:            Ground truth label of the video(s) used to generate the predictions
            :predictions:      The output predictions from the model accross all batches
            :name:             The name(s) of the video(s) currently being classified
        Return:
            :current_accuracy: The current classification accuracy of all videos
                               passed through this object accross multiple calls of this method
        """
        self.step = step
        self._save_prediction(label, predictions, names)

        if self.method == 'default':
            prediction = np.mean(predictions, 0).argmax()
            if prediction == label:
                self.correct_predictions += 1

            self.total_predictions += 1
            current_accuracy = self.get_accuracy()

        elif self.method == 'svm':
            current_accuracy = -1

        else:
            print "Error: Invalid classification method ", self.method
            exit()

        self.logger.add_scalar_value(os.path.join(self.log_name, 'acc'), current_accuracy, step=self.step)
        return current_accuracy


    def total_classification(self):
        """
        Args:
            :label:            Ground truth label of the video(s) used to generate the predictions
            :predictions:      The output predictions from the model accross all batches
            :name:             The name(s) of the video(s) currently being classified
        Return:
            :current_accuracy: The current classification accuracy of all videos
                               passed through this object accross multiple calls of this method
        """
        if self.method == 'default':
            accuracy = self._default_classify()

        elif self.method == 'svm':
            accuracy = self._svm_classify()

        else:
            print "Error: Invalid classification method ", self.method
            exit()

        self.logger.add_scalar_value(os.path.join(self.log_name, 'acc'), accuracy, step=self.step)

        return current_accuracy


    def _default_classify(self):
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
        self.clear_all()

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

        if len(model_output.shape)!=3:
            # Average together all clips relating to a video
            model_output = np.mean(model_output, axis=1)

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

        current_accuracy = self.correct_predictions / float(self.total_predictions)
        self.logger.add_scalar_value('test/acc',current_accuracy, step=self.total_predictions)
        return current_accuracy


    def _svm_classify(self):
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

        if len(model_output.shape)!=3:
            # Average together all clips relating to a video
            model_output = np.mean(model_output, axis=1)

        classifier = svm.SVC(kernel='linear')

        classifier.fit(model_output, labels)

        predictions = classifier.predict(model_output)

        # END IF

        for prediction_ind in range(len(predictions)):
            prediction = predictions[prediction_ind]
            label = labels[prediction_ind]
            name = names[prediction_ind]

            if self.verbose:
                print "vidName: ",name
                print "label:  ", label
                print "prediction: ", prediction

            self.predictions_array.append((prediction, label))
            self.total_predictions += 1
            if int(prediction) == int(label):
                self.correct_predictions += 1

        shutil.rmtree('temp')

        current_accuracy = self.correct_predictions / float(self.total_predictions)
        return current_accuracy


    def _save_prediction(self, label, prediction, name):
        if not os.path.isdir('temp'):
            os.mkdir('temp')

        np.save(name, (prediction, label))
