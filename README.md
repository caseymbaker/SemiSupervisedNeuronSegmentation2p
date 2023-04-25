# Semi-Supervised Deep Learning for Neuron Segmentation with Fewer Ground Truth Labels

Requirements: 
CUDA compatible GPU, Anaconda

## Demo on ABO 175 Data:

### Install Pipeline

1) Download repository and save to easily identifiable directory - e.g. *C:/Users/{username}/Documents/segmentation* 

2) Open Anaconda prompt and type:  

        cd segmentation_directory
        cd installation
        conda env create -f environment_suns.yml -n SSLseg 
        
  this uses Tensorflow V1, for Tensorflow V2, use environment_suns_tf2.yml
  
3) Go to the Anaconda environment folder, (e.g., C:/ProgramData/Anaconda3/envs or C:/Users/{username}/.conda/envs), and then go to folder `suns/Lib/site-packages/fissa`, overwrite core.py with the files provided in the installation folder. The modified files increase speed by eliminating redundant separate or separation_prep during initializating an Experiment object, and enable videos whose size are larger than 4 GB after converting to float32. If neither of them is important to you, then you can skip replacing the files. If you see a lot of text output when activating suns environment and do not want to see them, you can go to the Anaconda environment folder, go to folder `suns/etc/conda/activate.d`, and delete the two files under this folder. 

### Download ABO Data

4) The ABO dataset is available in [Allen Institute](https://github.com/AllenInstitute/AllenSDK/wiki/Use-the-Allen-Brain-Observatory-%E2%80%93-Visual-Coding-on-AWS). You may need a Amazon AWS account to download them. We used 10 videos from 175 um layer, {'501271265', '501704220', '501836392', '502115959', '502205092', '504637623', '510514474', '510517131', '540684467', '545446482'}. We used the manual labels [175 um layer](https://github.com/soltanianzadeh/STNeuroNet/tree/master/Markings/ABO/Layer175/FinalGT) created by Soltanian-Zadeh et al. We also used the code [create_h5_video_ABO.m](utils/create_h5_video_ABO.m) modified from the same STNeuroNet repository to crop each video to the first 20% durations and the center parts, so that the video sizes are changed from 512 x 512 x ~115,000 to 487 x 487 x ~23,000. Set the folders correctly, and run the code with `layer = 175`. 

5) Move the cropped video files into `demo/datanoSF/`

### Run Demo

6) In Anaconda Prompt, 

        cd segmentation_directory
        cd demo
        cd train_1_test_9
        conda activate SSLseg
        runABO.bat

### Pipeline Output

SNR videos can be found in `\demo\datanoSF\noSF\network_input`

10 frames of input can be found in `\demo\datanoSF\noSF\SelectedMasks\*_train.h5`

Model weights can be found in `\demo\datanoSF\noSF\Weights 1to9`

Output Masks, hyperparameters, and F1/Recall/Precision can be found in `\demo\datanoSF\noSF\output_masks 1to9`








