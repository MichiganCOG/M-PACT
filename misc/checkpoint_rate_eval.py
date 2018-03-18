import os
import numpy as np
# from sklearn.metrics import confusion_matrix
# import os
import matplotlib.pyplot as plt
# import scipy.misc as scm


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
    if trecs == 'base':
        fileind = 0
    elif trecs == 'static':
        fileind = 1
    elif trecs == 'dynamic':
        fileind = 2
    else:
        print "Not a valid preprocessing method"
        exit()

    loaded_files = []
    if modelName.upper() == 'C3D':
        fn = c3d_filenames
        for check in fn[fileind][:-1]:
            loaded_files.append(np.load(fn[fileind][-1].replace('CHECKVAL', str(check))))

    elif modelName.upper() == 'I3D':
        fn = i3d_filenames
        for check in fn[fileind][:-1]:
            loaded_files.append(np.load(fn[fileind][-1].replace('CHECKVAL', str(check))))

    elif modelName.upper() == 'TSN':
        fn = tsn_filenames
        for check in fn[fileind][:-1]:
            loaded_files.append(np.load(fn[fileind][-1].replace('CHECKVAL', str(check))))

    elif modelName.upper() == 'RESNET50 + LSTM':
        fn = resnet_filenames
        for check in fn[fileind][:-1]:
            loaded_files.append(np.load(fn[fileind][-1].replace('CHECKVAL', str(check))))

    return loaded_files, fn[fileind][:-1]

def get_rate(vidName, rates):
    # for i in range(len(rates)):
    #     if 'Bush_Wave_vs__' in rates[i][0]:
    #         print i , rates[i][0], rates[i][1]
    # import pdb;pdb.set_trace()
    try:
        if vidName[0] == '\'':
            if vidName[-1] != '\n':
                return float(rates[np.where(rates[:,0]==vidName[1:])][0][1])
            else:
                return float(rates[np.where(rates[:,0]==vidName[1:-1])][0][1])
        else:
            if vidName[-1] != '\n':
                return float(rates[np.where(rates[:,0]==vidName)][0][1])
            else:
                return float(rates[np.where(rates[:,0]==vidName[:-1])][0][1])
    except:
        print "Get rate faild for vid: ", vidName
        import pdb; pdb.set_trace()


def get_rates_by_bin(rates):
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

    return rate0, rate1, rate2, rate3


def gen_plot(totaccr, xLabel, labels, modelName, trecs, savetrecs, legend=True, legendSize=20):

    fig = plt.figure(num=1, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)

    # plt.plot(labels, totaccr[0], ls=(0,(5,10)), label=str(xLabel[0]))
    # plt.plot(labels, totaccr[1], ls='--', label=str(xLabel[1]))
    # plt.plot(labels, totaccr[2], ls='-.', label=str(xLabel[2]))
    # plt.plot(labels, totaccr[3], ls='-', label=str(xLabel[3]))
    #plt.plot(x, totaccr[4], ls=(0,(5,10)), label=str(labels[4]))
    plt.plot(labels, totaccr[0], '-', marker='o', markersize=20, linewidth=4, label=str(xLabel[0]), color='#db9e29')
    plt.plot(labels, totaccr[1], '-', marker='^', markersize=20, linewidth=4,label=str(xLabel[1]), color='#328717')
    plt.plot(labels, totaccr[2], '-', marker='h', markersize=20, linewidth=4,label=str(xLabel[2]), color='#0c78df')
    plt.plot(labels, totaccr[3], '-', marker='*', markersize=20, linewidth=4,label=str(xLabel[3]), color='#df0c0f')
    #plt.plot(alphas, acc_values[4], '-', marker='s', markersize=10, linewidth=2,label='CVR 2.5', color='#610cdf')

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






modelName = 'TSN'
trecs = 'base'
#trecs = 'static'
#trecs = 'dynamic'

def run_all(modelName, trecs, legend=True, legendSize=20):

    if trecs == 'static':
        if 'I3D' in modelName.upper():
            method = 'CVR 0.4'
        else:
            method = 'CVR 0.8'
    elif trecs == 'dynamic':
        if 'I3D' in modelName.upper():
            method = 'SR'
        else:
            method = 'RR'
    else:
        method = 'Baseline'


    print 'loading names and rates'

    rates_HMDB51 = np.load('/z/home/erichof/CVPR2018_data/testHMDB51Rate.npy')
    vid_names = np.load('../scripts/PBS/c3d/test_rate/vidNames.npy')

    print "loaded vidnames and rates"
    rates = []
    for name in vid_names:
        rates.append(get_rate(name+'.avi', rates_HMDB51))
    print "associated rates to vidnames"

    outputs, labels = load_filenames(modelName, trecs)
    print 'loaded outputs'

    # output1 = np.load('../results/c3d/HMDB51Rate/c3d_tf_HMDB51_bgr/checkpoint_2551_rate/test_predictions_HMDB51Rate_avg_pooling.npy')
    # output2 = np.load('../results/c3d/HMDB51Rate/c3d_tf_HMDB51_bgr/checkpoint_5101_rate/test_predictions_HMDB51Rate_avg_pooling.npy')
    # output3 = np.load('../results/c3d/HMDB51Rate/c3d_tf_HMDB51_bgr/checkpoint_7651_rate/test_predictions_HMDB51Rate_avg_pooling.npy')
    # output4 = np.load('../results/c3d/HMDB51Rate/c3d_tf_HMDB51_bgr/checkpoint_10201_rate/test_predictions_HMDB51Rate_avg_pooling.npy')
    # output5 = np.load('../results/c3d/HMDB51Rate/c3d_tf_HMDB51_bgr/checkpoint_12750_rate/test_predictions_HMDB51Rate_avg_pooling.npy')

    r0, r1, r2, r3 = get_rates_by_bin(rates)
    ros = []
    for output in outputs:
        ros.append([output[r0], output[r1], output[r2], output[r3]])
    # ro1 = [output1[r0], output1[r1], output1[r2], output1[r3]]
    # ro2 = [output2[r0], output2[r1], output2[r2], output2[r3]]
    # ro3 = [output3[r0], output3[r1], output3[r2], output3[r3]]
    # ro4 = [output4[r0], output4[r1], output4[r2], output4[r3]]
    # ro5 = [output5[r0], output5[r1], output5[r2], output5[r3]]
    print 'converted to rate bins'


    # acc1r = []
    # acc2r = []
    # acc3r = []
    # acc4r = []
    # acc5r = []

    accrs = []
    accs = []

    # acc1 = []
    # acc2 = []
    # acc3 = []
    # acc4 = []
    # acc5 = []

    for output in outputs:
        accs.append([float(len(np.where(output[:,0] == output[:,1])[0]))/len(output)])

    # acc1.append(float(len(np.where(output1[:,0] == output1[:,1])[0]))/len(output1))
    # acc2.append(float(len(np.where(output2[:,0] == output2[:,1])[0]))/len(output2))
    # acc3.append(float(len(np.where(output3[:,0] == output3[:,1])[0]))/len(output3))
    # acc4.append(float(len(np.where(output4[:,0] == output4[:,1])[0]))/len(output4))
    # acc5.append(float(len(np.where(output5[:,0] == output5[:,1])[0]))/len(output5))
    for ro in ros:
        taccr = []
        for i in range(4):
            taccr.append(float(len(np.where(ro[i][:,0] == ro[i][:,1])[0]))/len(ro[i]))
        accrs.append(taccr)

            # acc1r.append(float(len(np.where(ro1[i][:,0] == ro1[i][:,1])[0]))/len(ro1[i]))
            # acc2r.append(float(len(np.where(ro2[i][:,0] == ro2[i][:,1])[0]))/len(ro2[i]))
            # acc3r.append(float(len(np.where(ro3[i][:,0] == ro3[i][:,1])[0]))/len(ro3[i]))
            # acc4r.append(float(len(np.where(ro4[i][:,0] == ro4[i][:,1])[0]))/len(ro4[i]))
            # acc5r.append(float(len(np.where(ro5[i][:,0] == ro5[i][:,1])[0]))/len(ro5[i]))

    print "extracted accuracies"

    #labels = [2551, 5101, 7651, 10201, 12750]
    x = [0,1,2,3]
    xLabel = ['[0.2-0.6]', '[0.6-1.0]', '[1.0-2.0]', '[2.0-3.0]']
    totaccr = np.array(accrs).T
    #plt.xticks(x, label)
    #ax.grid(axis='x', linestyle='dashed')

    #import pdb; pdb.set_trace()
    gen_plot(totaccr, xLabel, labels, modelName, method, trecs, legend, legendSize)

modelName = 'TSN'
trecs = 'base'
#trecs = 'static'
#trecs = 'dynamic'
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
# fig = plt.figure(num=1, figsize=(13, 13), dpi=80, facecolor='w', edgecolor='k')
# ax = fig.add_subplot(111)
# fig.subplots_adjust(right=0.2)
# plt.plot(labels, totaccr[0], ls=(0,(5,10)), label=str(xLabel[0]))
# plt.plot(labels, totaccr[1], ls='--', label=str(xLabel[1]))
# plt.plot(labels, totaccr[2], ls='-.', label=str(xLabel[2]))
# plt.plot(labels, totaccr[3], ls='-', label=str(xLabel[3]))
# #plt.plot(x, totaccr[4], ls=(0,(5,10)), label=str(labels[4]))
# bbox_anch = (1.308,1.019)
# plt.legend(bbox_to_anchor=bbox_anch, numpoints=1)
# #lgd = ax.legend(loc=9, bbox_to_anchor=(0.5,-0.02))
# plt.show()
#
# #plt.savefig('testfig.pdf')


import pdb; pdb.set_trace()
