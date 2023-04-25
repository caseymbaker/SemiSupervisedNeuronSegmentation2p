REM Prepare for FFT-based spatial filtering
python learn_wisdom_ABO.py
python learn_wisdom_2d_ABO.py

REM leave-one-out cross validation
REM Training pipeline. The post-processing will slow down after running many times, 
REM so I split the parameter search into two scripts.

REM train-1-test-9 cross validation
REM Training pipeline
python train_CNN_params_ABO_1to9_noSF.py

REM Run SUNS batch. The post-processing will slow down after running many times, 
REM so I split each cross-validation into different scripts.
python test_batch_ABO_1to9_noSF.py 0
python test_batch_ABO_1to9_noSF.py 1
python test_batch_ABO_1to9_noSF.py 2
python test_batch_ABO_1to9_noSF.py 3
python test_batch_ABO_1to9_noSF.py 4
python test_batch_ABO_1to9_noSF.py 5
python test_batch_ABO_1to9_noSF.py 6
python test_batch_ABO_1to9_noSF.py 7
python test_batch_ABO_1to9_noSF.py 8
python test_batch_ABO_1to9_noSF.py 9
