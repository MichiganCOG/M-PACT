import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', action='store', required=True,
        help = 'Name of the model to be created, capitalization is recommended.')

args = parser.parse_args()


def make_directory(model):
    """
    Make directory to store the model and preprocessing files and add an __init__.py file to allow the model to be imported
    Arguments:
        :model: Name of the model to create a folder for
    """
    try:
        os.mkdir(os.path.join('models', model))
        init_file = open(os.path.join('models', model, '__init__.py'), 'w')
        init_file.write('')
        init_file.close()

    except:
        print "ERROR: That model folder already exists, please choose a different name or delete the folder."
        exit()

    # END TRY

def gen_model_file(model):
    """
    Generate the basic file structure for the model off of models/template_model.py
    The generated model file must then be completed manually

    Arguments:
        :model: Name of the model to create a model file for
    """
    template = open('models/template_model.py','r')
    contents = template.read()
    template.close()

    modified_contents = contents.replace('MODELNAME', model)

    output = open(os.path.join('models', model.lower(), model.lower()+'_model.py'), 'w')
    output.write(modified_contents)
    output.close()

def gen_preprocessing_file(model):
    """
    Generate the basic file structure for preprocessing off of models/template_preprocessing.py
    The generated preprocessing file must then be completed manually

    Arguments:
        :model: Name of the model to create a preprocessing file for
    """
    template = open('models/template_preprocessing.py','r')
    contents = template.read()
    template.close()

    output = open(os.path.join('models', model, 'default_preprocessing.py'), 'w')
    output.write(contents)
    output.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', action='store', required=True,
            help = 'Name of the model to be created, capitalization is recommended.')

    args = parser.parse_args()

    model_name = args.model_name
    model_low = model_name.lower()


    # Make directory for the model under models/model_low
    make_directory(model_low)

    # Generate default model file in the created directory, must be filled in manually
    gen_model_file(model_name)

    # Generate default preprocessing file in the created directory, must be filled in manually
    gen_preprocessing_file(model_low)
