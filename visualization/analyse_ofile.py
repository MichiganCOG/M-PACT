import csv
import itertools

import numpy             as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# Global variables
ofile        = open('/z/home/madantrg/tf-activity-recognition-framework/scripts/PBS/i3d/test/MIT_Mini/TFRecords_1gpu_I3D_Train_MIT_Mini_Test_MIT_Mini_Avg_Pooling_250.o82119').readlines()
#ofile        = open('/z/home/madantrg/tf-activity-recognition-framework/scripts/PBS/c3d/test/MIT_Mini/TFRecords_1gpu_C3D_Train_MIT_Mini_Test_MIT_Mini.o82125').readlines()
#ofile        = open('/z/home/madantrg/tf-activity-recognition-framework/scripts/PBS/tsn/test/MIT_Mini/TFRecords_1gpu_TSN_Train_MIT_Mini_Test_MIT_Mini.o82124').readlines()

lfile        = open('/z/dat/Moments_in_Time_Mini/moments_categories.txt').readlines()

def plot_confusion_matrix(true_labels, pred_labels, 
                          classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """ Plots confusion matrix
    Args:
        true_labels: List of true labels 
        pred_labels: List of predicted labels
        classes:     List of class names
        normalize:   Boolean indicating normalization of confusion matrix values
        title:       Title of confusion matrix plot
        cmap:        Matplotlib color map
    Returns:
        Nothing 
    """

    cm = confusion_matrix(true_labels, pred_labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")

    else:
        print('Confusion matrix, without normalization')

    # END IF

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_classwise_recog_acc(true_labels, pred_labels, output_dims, class_labels, delta=50):
    """ Plots classwise recognition accuracy across the entire dataset
    Args:
        true_labels:  List of true labels 
        pred_labels:  List of predicted labels
        class_labels: List of class names
        output_dims:  Number of output dimensions 
        delta:        Integer indicating number of classes to display in a figure
    Returns:
        Nothing 
    """

    pred_dict  = {}
    total_dict = {}

    # Dictionary initializer
    for item in range(output_dims):
        pred_dict[str(item)]  = 0
        total_dict[str(item)] = 0
   
    # END FOR
 
    # Update dictionary element counts
    for item in range(len(true_labels)):
        total_dict[str(true_labels[item])] += 1
        
        if true_labels[item] == pred_labels[item]:
            pred_dict[str(true_labels[item])] += 1

        # END IF

    # END FOR

    # Verification variables
    total      = 0
    corr_total = 0

    plotx = []
    ploty = []

    for item in range(len(total_dict.keys())):

        plotx.append(item)
        ploty.append(float(pred_dict[str(item)])/total_dict[str(item)])

        # Verification variables
        total      +=total_dict[str(item)]    
        corr_total +=pred_dict[str(item)]    

    print total, corr_total

    for item in range(0, output_dims, delta):
        plt.gcf()
        plt.bar(plotx[item:item+delta], ploty[item:item+delta])
        plt.xticks(plotx[item:item+delta], class_labels[item:item+delta], rotation='vertical')
        title_str = 'Recognition accuracy for class between '+str(item)+':'+str(item + len(plotx[item:item+delta]))
        plt.title(title_str)
        plt.show()

    # END FOR

def print_alt_class(true_labels, pred_labels, class_labels, output_dims, k=3):
    """ Print topk alternative class predictions for every dataset 
    Args:
        true_labels:  List of true labels 
        pred_labels:  List of predicted labels
        class_labels: List of class names
        output_dims:  Number of output dimensions 
        k:            Number of alternative class predictions to display 
    Returns:
        Nothing 
    """

    cm = confusion_matrix(true_labels, pred_labels)

    for item in range(output_dims):
        val_not_class   = cm[item, :] 
        index_not_class = np.argsort(val_not_class)
   
        print "############################################" 
        print "Current class: ", class_labels[item]
        print "Top class prediction: ", class_labels[index_not_class[-1]]
        print "Top ",3,
        print "Alternative class predicted: ", [class_labels[X] for X in index_not_class[-1-k:-1]]
        print "############################################" 

        print "\n"

def alt_class(true_labels, pred_labels, class_labels, output_dims, k=3):
    """ Generate CSV file containing topk alternative class predictions for the entire dataset
    Args:
        true_labels:  List of true labels 
        pred_labels:  List of predicted labels
        class_labels: List of class names
        output_dims:  Number of output dimensions 
        k:            Number of alternative class predictions to display 
    Returns:
        Nothing 
    """

    cm = confusion_matrix(true_labels, pred_labels)

    with open('family_labels.csv','wb') as family_file:
        wr = csv.writer(family_file)

        for item in range(output_dims):
            val_not_class   = cm[item, :] 
            index_not_class = np.argsort(val_not_class)
  
            op_alt_classes = []
            op_alt_classes = index_not_class[-k:]
            op_alt_classes = [word for word in op_alt_classes if word!=item]

            if item in index_not_class[-k:]:
                op_alt_classes.insert(0, index_not_class[-k-1])

            # END IF
            
            op_alt_class_names = []

            for X in op_alt_classes:
                op_alt_class_names.append(class_labels[X])       

            # END FOR

            op_alt_classes = op_alt_class_names + op_alt_classes

            # END FOR
            
            wr.writerow(op_alt_classes)

    # END WITH
        

if __name__=='__main__':


    output_dims  = 200
    class_labels = []

    for label in lfile:
        class_labels.append(label.split(',')[0])

    # END FOR

    ################## Parsing output of .O File from required point onwards ###########

    data_index = 0

    for index in range(len(ofile)):
        if 'accuracy' in ofile[index]:
            data_index = index + 1

        # END IF

    # END FOR

    data = ofile[data_index]
    data = eval(data)

    #####################################################################################
    
    # Collect data into required format
    true_labels = []
    pred_labels = []

    for data_item in data:
        true_labels.append(int(data_item[1]))
        pred_labels.append(int(data_item[0]))

    # END FOR

    assert(len(true_labels) == len(pred_labels))

    # Analysis plots
    # 1. Classwise recognition accuracy
    #plot_classwise_recog_acc(true_labels, pred_labels, output_dims, class_labels, 50)

    # 2. Confusion matrix
    #plot_confusion_matrix(true_labels, pred_labels, class_labels, normalize=True, title='Normalized confusion matrix')
   
    # 3. Print top-k alternative class predictions apart from true class
    #print_alt_class(true_labels, pred_labels, class_labels, output_dims, 5)

    # 4. Generate simple family classes
    family_file = alt_class(true_labels, pred_labels, class_labels, output_dims, 5)
