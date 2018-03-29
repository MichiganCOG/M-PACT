import abc
ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})

# Abstract Class that is used when generating models.
class Abstract_Model_Class(ABC):

    def __init__(self, modelName, preprocMethod, inputDims, outputDims, expName, clipLength, numVids, numEpochs, batchSize, numClips, numGpus, train, modelAlpha, inputAlpha, dropoutRate, freeze, verbose):

        self.preproc_method = preprocMethod
        self.input_dims = inputDims
        self.output_dims = outputDims
        self.exp_name = expName
        self.clip_length = clipLength
        self.num_vids = numVids
        self.num_epochs = numEpochs
        self.batch_size = batchSize
        self.num_clips = numClips
        self.num_gpus = numGpus
        self.dropout_rate = dropoutRate
        self.freeze = freeze

        if train == 1.0:
            self.istraining = True

        else:
            self.istraining = False

        # END IF

        self.model_alpha = modelAlpha
        self.input_alpha = inputAlpha
        self.verbose = verbose
        self.name = modelName
        self.track_variables = {}

        if ((self.preproc_method == 'rr') or (self.preproc_method == 'sr')):
            self.store_alpha = True

        # END IF


        if verbose:
            print self.name + " Model Initialized"

        # END IF

    def inference(self):
        raise NotImplementedError('Method not implemented in the specified model: inference')

    def load_default_weights(self):
        #raise NotImplementedError('Method not implemented in the specified model: load_default_weights')
        return None

    def preprocess_tfrecords(self):
        raise NotImplementedError('Method not implemented in the specified model: preprocess_tfrecords')

    def add_track_variables(self, variable_name, variable):
        self.track_variables[variable_name] = variable

    def get_track_variables(self):
        """
        Allow for tracking of variables within a model other than the defined layers
        """
        return self.track_variables

    def loss(self, logibs, labels, loss_type='full_loss'):
        raise NotImplementedError('Method not implemented in the specified model: loss')
