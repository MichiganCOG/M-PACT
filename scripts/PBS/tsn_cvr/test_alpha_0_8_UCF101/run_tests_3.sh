module load numpy cuda cudnn gflags tensorflow/1.2.1 opencv/2.4.13/16.04 
export CUDA_VISIBLE_DEVICES="2"
python train_test_TFRecords_multigpu_model.py \
--model tsn_cvr  \
--dataset UCF101 \
--loadedDataset UCF101 \
--numGpus 1 \
--train 0 \
--load 1 \
--size 224 \
--inputDims 60 \
--outputDims 101 \
--batchSize 1 \
--seqLength 60 \
--expName tsn_init_cvr_0_8_ucf \
--numVids 3783 \
--split 1 \
--baseDataPath /z/dat \
--fName testlist \
--modelAlpha 0.8 \
--inputAlpha 2.2 \
--metricsDir input_alpha_2_2 \
--verbose 1 











































