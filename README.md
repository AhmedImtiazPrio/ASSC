# End-to-end Sleep Staging with Raw Single Channel EEG using Deep Residual ConvNets
#### Accepted in *IEEE BHI 2019*

***Abstract:*** Humans approximately spend a third of their life sleeping, which makes monitoring sleep an integral part of well-being. In this paper, a 34-layer deep residual ConvNet architecture for end-to-end sleep staging is proposed. The network takes raw single channel electroencephalogram (Fpz-Cz) signal as input and yields hypnogram annotations for each 30s segments as output. Experiments are carried out for two different scoring standards (5 and 6 stage classification) on the expanded PhysioNet Sleep-EDF dataset, which contains multi-source data from hospital and household polysomnography setups. The performance of the proposed network is compared with that of the state-of-the-art algorithms in patient independent validation tasks. The experimental results demonstrate the superiority of the proposed network compared to the best existing method, providing a relative improvement in epoch-wise average accuracy of 6.8% and 6.3% on the household data and multi-source data, respectively.

#### For Sleep Stage benchmarking check out the [benchmark](https://github.com/AhmedImtiazPrio/ASSC/tree/r1.0/benchmark) folder containing filenames for the *SC-task* & *RS-task*, detailed in the paper.

## Dataset Download:

Download the ***Physionet Sleep-EDF Expanded*** dataset by running the bash script on linux inside the ASSC root:
```
bash bulkdownload.sh
```
## Usage:

Download and create *.csv* files containing EEG Data and hypnogram annotations, place them in the *data* sub-folder. The *.csv* files should have ***rows = number of 30s data epochs***. Columns should have a size of 3003 and arranged as:

***data points (columns 1-3000) | hypnogram annotation | epoch ID | recording ID***

Recording IDs are renamed from 1-61. 

Use the train.py file to train the proposed Resnet-34 Architecture for end-to-end sleep staging. Specify the **.csv** channel file to use from the data and the number of sleep stages to use (5 or 6).
```
python train.py FpzCz.csv 5 --epochs 200 --batch_size 64
```

#### For issues contact sushmit0109@gmail.com
