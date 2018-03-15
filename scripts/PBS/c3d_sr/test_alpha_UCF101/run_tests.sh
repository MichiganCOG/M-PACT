pids=
python train_test_TFRecords_multigpu_model.py \
--model c3d_sr \
--numGpus 1 \
--dataset UCF101 \
--loadedDataset UCF101 \
--train 0 \
--load 1 \
--inputDims 16 \
--outputDims 101 \
--seqLength 1 \
--size 112  \
--expName c3d_sr_UCF101 \
--numClips 1 \
--clipLength 50 \
--clipOffset random \
--numVids 3783 \
--split 1 \
--baseDataPath /z/dat \
--fName testlist \
--verbose 1 \
--inputAlpha 0.2 \
--metricsDir input_alpha_0_2 & pids+=" $!"
wait $pids || { echo "there were esrors" >&2; exit 1; }

python train_test_TFRecords_multigpu_model.py \
--model c3d_sr \
--numGpus 1 \
--dataset UCF101 \
--loadedDataset UCF101 \
--train 0 \
--load 1 \
--inputDims 16 \
--outputDims 101 \
--seqLength 1 \
--size 112  \
--expName c3d_sr_UCF101 \
--numClips 1 \
--clipLength 50 \
--clipOffset random \
--numVids 3783 \
--split 1 \
--baseDataPath /z/dat \
--fName testlist \
--verbose 1 \
--inputAlpha 0.4 \
--metricsDir input_alpha_0_4 & pids+=" $!"
wait $pids || { echo "there were esrors" >&2; exit 1; }


python train_test_TFRecords_multigpu_model.py \
--model c3d_sr \
--numGpus 1 \
--dataset UCF101 \
--loadedDataset UCF101 \
--train 0 \
--load 1 \
--inputDims 16 \
--outputDims 101 \
--seqLength 1 \
--size 112  \
--expName c3d_sr_UCF101 \
--numClips 1 \
--clipLength 50 \
--clipOffset random \
--numVids 3783 \
--split 1 \
--baseDataPath /z/dat \
--fName testlist \
--verbose 1 \
--inputAlpha 0.6 \
--metricsDir input_alpha_0_6 & pids+=" $!"
wait $pids || { echo "there were esrors" >&2; exit 1; }


python train_test_TFRecords_multigpu_model.py \
--model c3d_sr \
--numGpus 1 \
--dataset UCF101 \
--loadedDataset UCF101 \
--train 0 \
--load 1 \
--inputDims 16 \
--outputDims 101 \
--seqLength 1 \
--size 112  \
--expName c3d_sr_UCF101 \
--numClips 1 \
--clipLength 50 \
--clipOffset random \
--numVids 3783 \
--split 1 \
--baseDataPath /z/dat \
--fName testlist \
--verbose 1 \
--inputAlpha 0.8 \
--metricsDir input_alpha_0_8 & pids+=" $!"
wait $pids || { echo "there were esrors" >&2; exit 1; }


python train_test_TFRecords_multigpu_model.py \
--model c3d_sr \
--numGpus 1 \
--dataset UCF101 \
--loadedDataset UCF101 \
--train 0 \
--load 1 \
--inputDims 16 \
--outputDims 101 \
--seqLength 1 \
--size 112  \
--expName c3d_sr_UCF101 \
--numClips 1 \
--clipLength 50 \
--clipOffset random \
--numVids 3783 \
--split 1 \
--baseDataPath /z/dat \
--fName testlist \
--verbose 1 \
--inputAlpha 1.0 \
--metricsDir input_alpha_1_0 & pids+=" $!"
wait $pids || { echo "there were esrors" >&2; exit 1; }


python train_test_TFRecords_multigpu_model.py \
--model c3d_sr \
--numGpus 1 \
--dataset UCF101 \
--loadedDataset UCF101 \
--train 0 \
--load 1 \
--inputDims 16 \
--outputDims 101 \
--seqLength 1 \
--size 112  \
--expName c3d_sr_UCF101 \
--numClips 1 \
--clipLength 50 \
--clipOffset random \
--numVids 3783 \
--split 1 \
--baseDataPath /z/dat \
--fName testlist \
--verbose 1 \
--inputAlpha 1.2 \
--metricsDir input_alpha_1_2 & pids+=" $!"
wait $pids || { echo "there were esrors" >&2; exit 1; }


python train_test_TFRecords_multigpu_model.py \
--model c3d_sr \
--numGpus 1 \
--dataset UCF101 \
--loadedDataset UCF101 \
--train 0 \
--load 1 \
--inputDims 16 \
--outputDims 101 \
--seqLength 1 \
--size 112  \
--expName c3d_sr_UCF101 \
--numClips 1 \
--clipLength 50 \
--clipOffset random \
--numVids 3783 \
--split 1 \
--baseDataPath /z/dat \
--fName testlist \
--verbose 1 \
--inputAlpha 1.4 \
--metricsDir input_alpha_1_4 & pids+=" $!"
wait $pids || { echo "there were esrors" >&2; exit 1; }
python train_test_TFRecords_multigpu_model.py \
--model c3d_sr \
--numGpus 1 \
--dataset UCF101 \
--loadedDataset UCF101 \
--train 0 \
--load 1 \
--inputDims 16 \
--outputDims 101 \
--seqLength 1 \
--size 112  \
--expName c3d_sr_UCF101 \
--numClips 1 \
--clipLength 50 \
--clipOffset random \
--numVids 3783 \
--split 1 \
--baseDataPath /z/dat \
--fName testlist \
--verbose 1 \
--inputAlpha 1.6 \
--metricsDir input_alpha_1_6 & pids+=" $!"
wait $pids || { echo "there were esrors" >&2; exit 1; }
python train_test_TFRecords_multigpu_model.py \
--model c3d_sr \
--numGpus 1 \
--dataset UCF101 \
--loadedDataset UCF101 \
--train 0 \
--load 1 \
--inputDims 16 \
--outputDims 101 \
--seqLength 1 \
--size 112  \
--expName c3d_sr_UCF101 \
--numClips 1 \
--clipLength 50 \
--clipOffset random \
--numVids 3783 \
--split 1 \
--baseDataPath /z/dat \
--fName testlist \
--verbose 1 \
--inputAlpha 1.8 \
--metricsDir input_alpha_1_8 & pids+=" $!"
wait $pids || { echo "there were esrors" >&2; exit 1; }
python train_test_TFRecords_multigpu_model.py \
--model c3d_sr \
--numGpus 1 \
--dataset UCF101 \
--loadedDataset UCF101 \
--train 0 \
--load 1 \
--inputDims 16 \
--outputDims 101 \
--seqLength 1 \
--size 112  \
--expName c3d_sr_UCF101 \
--numClips 1 \
--clipLength 50 \
--clipOffset random \
--numVids 3783 \
--split 1 \
--baseDataPath /z/dat \
--fName testlist \
--verbose 1 \
--inputAlpha 2.0 \
--metricsDir input_alpha_2_0 & pids+=" $!"
wait $pids || { echo "there were esrors" >&2; exit 1; }

python train_test_TFRecords_multigpu_model.py \
--model c3d_sr \
--numGpus 1 \
--dataset UCF101 \
--loadedDataset UCF101 \
--train 0 \
--load 1 \
--inputDims 16 \
--outputDims 101 \
--seqLength 1 \
--size 112  \
--expName c3d_sr_UCF101 \
--numClips 1 \
--clipLength 50 \
--clipOffset random \
--numVids 3783 \
--split 1 \
--baseDataPath /z/dat \
--fName testlist \
--verbose 1 \
--inputAlpha 2.2 \
--metricsDir input_alpha_2_2 & pids+=" $!"
wait $pids || { echo "there were esrors" >&2; exit 1; }

python train_test_TFRecords_multigpu_model.py \
--model c3d_sr \
--numGpus 1 \
--dataset UCF101 \
--loadedDataset UCF101 \
--train 0 \
--load 1 \
--inputDims 16 \
--outputDims 101 \
--seqLength 1 \
--size 112  \
--expName c3d_sr_UCF101 \
--numClips 1 \
--clipLength 50 \
--clipOffset random \
--numVids 3783 \
--split 1 \
--baseDataPath /z/dat \
--fName testlist \
--verbose 1 \
--inputAlpha 2.4 \
--metricsDir input_alpha_2_4 & pids+=" $!"
wait $pids || { echo "there were esrors" >&2; exit 1; }

python train_test_TFRecords_multigpu_model.py \
--model c3d_sr \
--numGpus 1 \
--dataset UCF101 \
--loadedDataset UCF101 \
--train 0 \
--load 1 \
--inputDims 16 \
--outputDims 101 \
--seqLength 1 \
--size 112  \
--expName c3d_sr_UCF101 \
--numClips 1 \
--clipLength 50 \
--clipOffset random \
--numVids 3783 \
--split 1 \
--baseDataPath /z/dat \
--fName testlist \
--verbose 1 \
--inputAlpha 2.6 \
--metricsDir input_alpha_2_6 & pids+=" $!"
wait $pids || { echo "there were esrors" >&2; exit 1; }

python train_test_TFRecords_multigpu_model.py \
--model c3d_sr \
--numGpus 1 \
--dataset UCF101 \
--loadedDataset UCF101 \
--train 0 \
--load 1 \
--inputDims 16 \
--outputDims 101 \
--seqLength 1 \
--size 112  \
--expName c3d_sr_UCF101 \
--numClips 1 \
--clipLength 50 \
--clipOffset random \
--numVids 3783 \
--split 1 \
--baseDataPath /z/dat \
--fName testlist \
--verbose 1 \
--inputAlpha 2.8 \
--metricsDir input_alpha_2_8 & pids+=" $!"
wait $pids || { echo "there were esrors" >&2; exit 1; }

python train_test_TFRecords_multigpu_model.py \
--model c3d_sr \
--numGpus 1 \
--dataset UCF101 \
--loadedDataset UCF101 \
--train 0 \
--load 1 \
--inputDims 16 \
--outputDims 101 \
--seqLength 1 \
--size 112  \
--expName c3d_sr_UCF101 \
--numClips 1 \
--clipLength 50 \
--clipOffset random \
--numVids 3783 \
--split 1 \
--baseDataPath /z/dat \
--fName testlist \
--verbose 1 \
--inputAlpha 3.0 \
--metricsDir input_alpha_3_0 & pids+=" $!"
wait $pids || { echo "there were esrors" >&2; exit 1; }
