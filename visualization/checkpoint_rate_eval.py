import os

import numpy             as np
import matplotlib.pyplot as plt

# Checkpoint numbers and locations for each model and its variants

c3d_filenames =    [[4463, 8926, 13388, 17851, 22313, '/z/home/madantrg/tf-activity-recognition-framework/scripts/PBS/c3d/test_rate/C3D_Rate_CHECKVAL.npy'],
                    [4463, 8926, 13388, 17851, 22313, '/z/home/madantrg/tf-activity-recognition-framework/scripts/PBS/c3d_cvr/test_rate_0_8/C3D_CVR_0_8_Rate_CHECKVAL.npy'],
                    [4463, 8926, 13388, 17851, 22313, '/z/home/madantrg/tf-activity-recognition-framework/scripts/PBS/c3d_rr/test_rate/C3D_RR_Rate_CHECKVAL.npy']]


i3d_filenames =    [[2678, 5356, 8033, 10711, 13388, '/z/home/madantrg/tf-activity-recognition-framework/scripts/PBS/i3d/test_rate/I3D_Rate_CHECKVAL.npy'],
                    [2678, 5356, 8033, 10711, 13388, '/z/home/madantrg/tf-activity-recognition-framework/scripts/PBS/i3d_cvr/test_rate/I3D_CVR_0_4_Rate_CHECKVAL.npy'],
                    [2678, 5356, 8033, 10711, 13388, '/z/home/madantrg/tf-activity-recognition-framework/scripts/PBS/i3d_sr/test_rate/I3D_SR_Rate_CHECKVAL.npy']]

resnet_filenames = [[2678, 5356, 8033, 10711, 13388, '/z/home/madantrg/tf-activity-recognition-framework/scripts/PBS/resnet_offset_fixed/test_rate/Resnet_Rate_CHECKVAL.npy'],
                    [2678, 5356, 8033, 10711, 13388, '/z/home/madantrg/tf-activity-recognition-framework/scripts/PBS/resnet_cvr/test_rate/Resnet_CVR_0_8_Rate_CHECKVAL.npy'],
                    [2678, 5356, 8033, 10711, 13388, '/z/home/madantrg/tf-activity-recognition-framework/scripts/PBS/resnet_rr/test_rate/Resnet_RR_Rate_CHECKVAL.npy']]

tsn_filenames =    [[2678, 4463, 7141, 8926, 11158, '/z/home/madantrg/tf-activity-recognition-framework/scripts/PBS/tsn/test_rate/TSN_Rate_CHECKVAL.npy'],
                    [2678, 4463, 7141, 8926, 11158, '/z/home/madantrg/tf-activity-recognition-framework/scripts/PBS/tsn_cvr/test_rate/TSN_CVR_0_8_Rate_CHECKVAL.npy'],
                    [2678, 4463, 7141, 8926, 11158, '/z/home/madantrg/tf-activity-recognition-framework/scripts/PBS/tsn_rr/test_rate/TSN_RR_Rate_CHECKVAL.npy']]

def load_filenames(modelName, trecs):
    """
    Load data and filenames for each model specified 
    Args:
        :modelName: Name of the base model 
        :trecs:     Type of base model TRECS variant (base, static or dynamic)

    Returns:
        Data loaded from desired output numpy file and checkpoints 
    """

    if trecs == 'base':
        fileind = 0

    elif trecs == 'static':
        fileind = 1

    elif trecs == 'dynamic':
        fileind = 2

    else:
        print "Not a valid preprocessing method"
        exit()

    # END IF

    loaded_files = []

    if modelName.upper() == 'C3D':
        fn = c3d_filenames

        for check in fn[fileind][:-1]:
            loaded_files.append(np.load(fn[fileind][-1].replace('CHECKVAL', str(check))))

        # END FOR

    elif modelName.upper() == 'I3D':
        fn = i3d_filenames

        for check in fn[fileind][:-1]:
            loaded_files.append(np.load(fn[fileind][-1].replace('CHECKVAL', str(check))))

        # END FOR

    elif modelName.upper() == 'TSN':
        fn = tsn_filenames

        for check in fn[fileind][:-1]:
            loaded_files.append(np.load(fn[fileind][-1].replace('CHECKVAL', str(check))))

        # END FOR

    elif modelName.upper() == 'RESNET50 + LSTM':
        fn = resnet_filenames

        for check in fn[fileind][:-1]:
            loaded_files.append(np.load(fn[fileind][-1].replace('CHECKVAL', str(check))))

        # END FOR

    # END IF

    return loaded_files, fn[fileind][:-1]

def get_rate(vidName, rates):
    """
    Return the rate used to modify the original video 
    Args:
        :vidName: Name of current video being processed
        :rates:   Numpy array containing the rates used to modify the original video to obtain RateModified dataset

    Returns:
       Return the rate used to modify the original video to obtain the current specified video 
    """

    try:
        if vidName[0] == '\'':
            if vidName[-1] != '\n':
                return float(rates[np.where(rates[:,0]==vidName[1:])][0][1])

            else:
                return float(rates[np.where(rates[:,0]==vidName[1:-1])][0][1])

            # END IF

        else:
            if vidName[-1] != '\n':
                return float(rates[np.where(rates[:,0]==vidName)][0][1])

            else:
                return float(rates[np.where(rates[:,0]==vidName[:-1])][0][1])

            # END IF

    except:
        print "Get rate faild for vid: ", vidName
        import pdb; pdb.set_trace()

    # END TRY

def get_rates_by_bin(rates):
    """
    Divide the rates into desired bins 
    Args:
        :rates:   Numpy array containing the rates used to modify the original video to obtain RateModified dataset

    Returns:
       Return four lists containing the rates divided into bins
    """

    # 0.2 - 0.6
    rate0 = []

    # 0.6 - 1.0
    rate1 = []

    # 1.0 - 2.0
    rate2 = []

    # 2.0 - 3.0
    rate3 = []

    for ind in range(len(rates)):
        if rates[ind] > 2.0:
            rate3.append(ind)

        elif rates[ind] > 1.0:
            rate2.append(ind)

        elif rates[ind] > 0.6:
            rate1.append(ind)

        else:
            rate0.append(ind)

        # END IF

    # END FOR

    return rate0, rate1, rate2, rate3


def gen_plot(totaccr, xLabel, labels, modelName, trecs, savetrecs, legend=True, legendSize=20):
    """
    Plot checkpoint results 
    Args:
        :totaccr:    Rate Bin-based accuracy
        :xLabel:     Legend labels for rate bins
        :labels:     Checkpoint numbers 
        :modelName:  Base model name 
        :trecs:      Fine category of TRECS models
        :savetrecs:  Overall categories Base, Static and Dynamic 
        :legend:     Binary value to display legend or not
        :legendSize: Size of legend in figure  

    Returns:
       Nothing 
    """

    fig = plt.figure(num=1, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')
    ax  = fig.add_subplot(111)

    plt.plot(labels, totaccr[0], '-', marker='o', markersize=20, linewidth=4, label=str(xLabel[0]), color='#db9e29')
    plt.plot(labels, totaccr[1], '-', marker='^', markersize=20, linewidth=4,label=str(xLabel[1]), color='#328717')
    plt.plot(labels, totaccr[2], '-', marker='h', markersize=20, linewidth=4,label=str(xLabel[2]), color='#0c78df')
    plt.plot(labels, totaccr[3], '-', marker='*', markersize=20, linewidth=4,label=str(xLabel[3]), color='#df0c0f')

    actual_title= modelName + ' HMDB51Rate Checkpoints ('+trecs+')'
    plt.title(actual_title, fontsize=35)
    plt.xlabel('Training Step', fontsize=25)
    plt.ylabel('Accuracy (%)', fontsize=25)

    if legend:
        plt.legend(fontsize=legendSize, loc='best')

    plt.tick_params(labelsize=15)

    ttl = ax.title
    ttl.set_position([0.5, 1.05])

    plt.tick_params(labelsize=25)
    plt.savefig(os.path.join('/z/home/erichof/Madan_TFRecords/checkpoint_figs', modelName + '_' + savetrecs + '.pdf'))
    #plt.show()
    plt.cla()
    plt.clf()
    plt.close()

def run_all(modelName, trecs, legend=True, legendSize=20):
    """
    Process data to generate checkpoint based accuracy plots on the rate-modified dataset (HMDB51Rate)     
    Args:
        :modelName:  Base model name 
        :trecs:      Coarse category of TRECS models
        :legend:     Binary value to display legend or not
        :legendSize: Size of legend in figure  

    Returns:
       Nothing 
    """

    if trecs == 'static':
        if 'I3D' in modelName.upper():
            method = 'CVR 0.4'

        else:
            method = 'CVR 0.8'

        # END IF

    elif trecs == 'dynamic':
        if 'I3D' in modelName.upper():
            method = 'SR'

        else:
            method = 'RR'

        # END IF

    else:
        method = 'Baseline'

    # END IF

    print 'Loading names and rates'

    rates_HMDB51 = np.load('/z/home/erichof/CVPR2018_data/testHMDB51Rate.npy')
    vid_names    = np.load('../scripts/PBS/c3d/test_rate/vidNames.npy')

    print "loaded vidnames and rates"
    rates = []

    for name in vid_names:
        rates.append(get_rate(name+'.avi', rates_HMDB51))

    # END FOR

    print "associated rates to vidnames"

    outputs, labels = load_filenames(modelName, trecs)
    print 'loaded outputs'

    r0, r1, r2, r3 = get_rates_by_bin(rates)
    ros = []

    for output in outputs:
        ros.append([output[r0], output[r1], output[r2], output[r3]])

    # END FOR

    print 'converted to rate bins'

    accrs = []
    accs  = []

    for output in outputs:
        accs.append([float(len(np.where(output[:,0] == output[:,1])[0]))/len(output)])

    # END FOR

    for ro in ros:
        taccr = []

        for i in range(4):
            taccr.append(float(len(np.where(ro[i][:,0] == ro[i][:,1])[0]))/len(ro[i]))

        # END FOR

        accrs.append(taccr)

    # END FOR

    print "extracted accuracies"

    x       = [0,1,2,3]
    xLabel  = ['[0.2-0.6]', '[0.6-1.0]', '[1.0-2.0]', '[2.0-3.0]']
    totaccr = np.array(accrs).T

    gen_plot(totaccr, xLabel, labels, modelName, method, trecs, legend, legendSize)


if __name__=="__main__":

    run_all('C3D', 'base')
    run_all('C3D', 'static')
    run_all('C3D', 'dynamic', legend=False)
    
    run_all('I3D', 'base', legendSize=15)
    run_all('I3D', 'static', legendSize=15)
    run_all('I3D', 'dynamic')
    
    run_all('TSN', 'base')
    run_all('TSN', 'static')
    run_all('TSN', 'dynamic')
    
    run_all('ResNet50 + LSTM', 'base')
    run_all('ResNet50 + LSTM', 'static')
    run_all('ResNet50 + LSTM', 'dynamic')
    
    import pdb; pdb.set_trace()
