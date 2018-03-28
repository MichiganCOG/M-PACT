import glob
import importlib


def create_model_object(**kwargs):
    """
    Use model_name to find a matching model class with that name
    All model classes are initialized from the same abstract class so just call that initializer

    Arguments:
    :kwargs: arguments specified in training and testing program

    Returns:
    :model:  model object initialized based off of the given model name
    """
    model_files = glob.glob('models/*/*_model.py')
    all_list = list()
    model_name = kwargs['modelName']
    for mf in model_files:
        module_name = mf[:-3]
        module_name = module_name.replace('/','.')
        module = importlib.import_module(module_name)
        module_lower = map(lambda module_x: module_x.lower(), dir(module))

        if model_name.lower() in module_lower:
            model_index = module_lower.index(model_name.lower())
            model = getattr(module, dir(module)[model_index])(**kwargs)
            return model

        # END IF

    # END FOR

    print "Model not found, specify model name in lowercase and try again. Ensure model is in a folder within 'models' directory and includes model file 'name_model.py'."
    exit()
