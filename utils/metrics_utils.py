"""
FILE TO CONTAIN VARIOUS METHODS OF CALCULATING PERFORMANCE METRICS FOR DIFFERENT MODELS
"""

import os
import shutil

import numpy      as np
import tensorflow as tf
from   sklearn    import svm


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
        :_avg_pooling_classify:
        :_last_frame_classify:
        :_svm_classify:
    """

    def __init__(self, output_dims, logger, method, is_training, model_name, exp_name, dataset, verbose=1):
        """
        Args:
            :output_dims: Output dimensions of the model, used to verify the shape of predictions
            :verbose:     Setting verbose command
        """
        self.output_dims=output_dims
        self.verbose=verbose
        self.model_name = model_name
        self.exp_name = exp_name
        self.dataset = dataset
        self.correct_predictions=0
        self.total_predictions=0
        self.predictions_array=[]
        self.logger=logger
        self.method=method
        self.step=0
        self.is_training = is_training
        self.file_name_dict = {}
        if self.is_training:
            self.log_name = 'train'

        else:
            self.log_name = 'test'

        # if os.path.isdir(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp')) and self.method != 'svm':
        #     shutil.rmtree(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp'))

        if self.method == 'svm':
            if not os.path.isdir(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp', 'svm_train')):
                print "\nError: Temporary training features are not present to train svm. Please first evaluate this model on the training split of this dataset using metricsMethod svm_train.\n"
                exit()


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

        if self.method == 'avg_pooling':
            if len(predictions.shape) >= 2:
                predictions = np.mean(predictions, 0)
            prediction = predictions.argmax()

        elif self.method == 'last_frame':
            if len(predictions.shape) >= 2:
                predictions = predictions[-1]
            prediction = predictions.argmax()

        elif 'svm' in self.method:
            prediction = -1

        else:
            print "Error: Invalid classification method ", self.method
            exit()


        if prediction == label:
            self.correct_predictions += 1

        self.total_predictions += 1
        current_accuracy = self.get_accuracy()

        if self.verbose:
            print "vidName: ",names
            print "label:  ", label
            print "prediction: ", prediction

        self.logger.add_scalar_value(os.path.join(self.log_name, 'acc_'+self.method), current_accuracy, step=self.step)
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
        if self.method == 'avg_pooling':
            accuracy = self._avg_pooling_classify()

        elif self.method == 'last_frame':
            accuracy = self._last_frame_classify()

        elif self.method == 'svm':
            accuracy = self._svm_classify()

        elif self.method == 'svm_train':
            print 'Please now classify this model using the testing split of this dataset.'
            accuracy = -1

        else:
            print "Error: Invalid classification method ", self.method
            exit()

        self.logger.add_scalar_value(os.path.join(self.log_name, 'acc_'+self.method), accuracy, step=self.step)

        return accuracy


    def _avg_pooling_classify(self):
        """
        Default argmax classification averaing the outputs of all frames
        Args:

        Return:
            :current_accuracy: The current classification accuracy of all videos
                               passed through this object accross multiple calls of this method
        """
        self.clear_all()

        model_output = []
        labels = []
        names = []

        # Load the saved model outputs from the temp folder storing each video as a new index in model_output and appending the outputs to that index
        for vid_name in os.listdir(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method)):
            for clip in os.listdir(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method, vid_name)):
                if vid_name in names:
                    ind = names.index(vid_name)
                    data = np.load(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method,vid_name, clip))
                    model_output[ind].append(data[0])

                else:
                    data = np.load(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method,vid_name, clip))
                    model_output.append([data[0]])
                    names.append(vid_name)
                    labels.append(data[1])

        # For each video, average the predictions within clips and frames therein then take the argmax prediction and compare it to the ground truth sabel
        for index in range(len(model_output)):
            model_output_dimensions = len(np.array(model_output[index]).shape)
            if model_output_dimensions > 2:
                model_output[index] = np.mean(model_output[index], axis=tuple(range(1,model_output_dimensions-1)) )   # Average everything except the dimensions for the number of clips and the outputs

            # Average the outputs for the clips
            model_output[index] = np.mean(model_output[index], 0)
            prediction = model_output[index].argmax()
            label = labels[index]

            if self.verbose:
                print "vidName: ",names[index]
                print "label:  ", label
                print "prediction: ", prediction

            self.predictions_array.append((prediction, label))
            self.total_predictions += 1
            if int(prediction) == int(label):
                self.correct_predictions += 1

            current_accuracy = self.correct_predictions / float(self.total_predictions)

        # END FOR

        shutil.rmtree(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp'))

        return current_accuracy


    def _last_frame_classify(self):
        """
        Classification based off of the last frame of each clip
        Args:

        Return:
            :current_accuracy: The current classification accuracy of all videos
                               passed through this object accross multiple calls of this method
        """
        self.clear_all()

        model_output = []
        labels = []
        names = []

        # Load the saved model outputs from the temp folder storing each video as a new index in model_output and appending the outputs to that index
        for vid_name in os.listdir(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method)):
            for clip in os.listdir(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method, vid_name)):
                if vid_name in names:
                    ind = names.index(vid_name)
                    data = np.load(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method,vid_name, clip))
                    model_output[ind].append(data[0])

                else:
                    data = np.load(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method,vid_name, clip))
                    model_output.append([data[0]])
                    names.append(vid_name)
                    labels.append(data[1])

        # For each video, select only the last frame of each clip and average the last frames then take the argmax prediction and compare it to the ground truth sabel
        for index in range(len(model_output)):
            model_output_dimensions = len(np.array(model_output[index]).shape)
            if model_output_dimensions > 2:
                model_output[index] = np.array(model_output[index])[:,-1,:]

            # Average the outputs for the clips
            model_output[index] = np.mean(model_output[index], 0)
            prediction = model_output[index].argmax()
            label = labels[index]

            if self.verbose:
                print "vidName: ",names[index]
                print "label:  ", label
                print "prediction: ", prediction

            self.predictions_array.append((prediction, label))
            self.total_predictions += 1
            if int(prediction) == int(label):
                self.correct_predictions += 1

            current_accuracy = self.correct_predictions / float(self.total_predictions)

        # END FOR

        shutil.rmtree(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp'))

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

        self.clear_all()

        model_output = []
        labels = []
        names = []

        # Load the saved model outputs from the temp folder storing each video as a new index in model_output and appending the outputs to that index
        for vid_name in os.listdir(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method+'_train')):
            for clip in os.listdir(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method+'_train', vid_name)):
                if vid_name in names:
                    ind = names.index(vid_name)
                    data = np.load(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method+'_train',vid_name, clip))
                    model_output[ind].append(data[0])

                else:
                    data = np.load(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method+'_train',vid_name, clip))
                    model_output.append([data[0]])
                    names.append(vid_name)
                    labels.append(data[1])

        # For each video, average the predictions within clips and frames therein then take the argmax prediction and compare it to the ground truth sabel
        for index in range(len(model_output)):
            model_output_dimensions = len(np.array(model_output[index]).shape)
            if model_output_dimensions > 2:
                model_output[index] = np.mean(model_output[index], axis=tuple(range(1,model_output_dimensions-1)) )   # Average everything except the dimensions for the number of clips and the outputs

            # Average the outputs for the clips
            model_output[index] = np.mean(model_output[index], 0)

        classifier = svm.SVC(kernel='linear')

        classifier.fit(model_output, labels)

        self.clear_all()

        model_output = []
        labels = []
        names = []

        # Load the saved model outputs from the temp folder storing each video as a new index in model_output and appending the outputs to that index
        for vid_name in os.listdir(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method)):
            for clip in os.listdir(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method, vid_name)):
                if vid_name in names:
                    ind = names.index(vid_name)
                    data = np.load(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method,vid_name, clip))
                    model_output[ind].append(data[0])

                else:
                    data = np.load(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method,vid_name, clip))
                    model_output.append([data[0]])
                    names.append(vid_name)
                    labels.append(data[1])

        # For each video, average the predictions within clips and frames therein then take the argmax prediction and compare it to the ground truth sabel
        for index in range(len(model_output)):
            model_output_dimensions = len(np.array(model_output[index]).shape)
            if model_output_dimensions > 2:
                model_output[index] = np.mean(model_output[index], axis=tuple(range(1,model_output_dimensions-1)) )   # Average everything except the dimensions for the number of clips and the outputs

            # Average the outputs for the clips
            model_output[index] = np.mean(model_output[index], 0)


        predictions = classifier.predict(model_output)

        for video in range(len(predictions)):
            prediction = predictions[video]
            label = labels[video]

            if self.verbose:
                print "vidName: ",names[video]
                print "label:  ", label
                print "prediction: ", prediction

            self.predictions_array.append((prediction, label))
            self.total_predictions += 1
            if int(prediction) == int(label):
                self.correct_predictions += 1

            current_accuracy = self.correct_predictions / float(self.total_predictions)

        # END FOR

        #shutil.rmtree(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp'))

        return current_accuracy




    def _save_prediction(self, label, prediction, name):
        if not os.path.isdir(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp')):
            os.mkdir(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp'))

        if not os.path.isdir(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp', self.method)):
            os.mkdir(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method))

        if not os.path.isdir(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method, name)):
            os.mkdir(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method, name))
            self.file_name_dict[name] = 0

        np.save(os.path.join('results', self.model_name, self.dataset, self.exp_name,'temp',self.method, name, str(self.file_name_dict[name])+'.npy'), (prediction, label))
        self.file_name_dict[name]+=1
