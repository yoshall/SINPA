# SINPA
This repo is the implementation of our IJCAI 2024 paper (AI for Social Good Track) entitled [Predicting Carpark Availability in Singapore with Cross-Domain Data: A New Dataset and A Data-Driven Approach](https://arxiv.org/pdf/2405.18910).
In this study, we crawl, process, and release the <b>SINPA</b> dataset, a large-scale parking availability dataset incorporating cross-domain data in Singapore. We then propose a novel deep-learning framework <b>DeepPA</b> to collectively forecast future PA readings across Singapore. 


## Framework
<img src="img/intro and model.png" width="1000px">

Figure (a) Distribution of 1,687 carparks throughout Singapore. (b) The framework of DeepPA.

## Dataset
In this section, we will outline the procedure for downloading the SINPA dataset, followed by a detailed description of the dataset. 
- **Dataset Download**. We provide the dataset on: [https://huggingface.co/datasets/Huaiwu/SINPA/tree/main](https://huggingface.co/datasets/Huaiwu/SINPA/tree/main). There are five files in the `./data` folder:
  ```
    ├── data
    │   ├── train.npz
    │   ├── val.npz
    │   ├── test.npz
  ```
    `train.npz`, `val.npz` and `test.npz` include training (12167 samples), validation(1217 samples), and test (1216 samples) set respectively. 
    To download the data, you can download all data from the provided [link](https://huggingface.co/datasets/Huaiwu/SINPA). You can download each file by clicking on its download button.

- **Dataset Description**. We crawled over three-year real-time PA data every 5 minutes from 1,921 parking lots throughout Singapore from [Data.gov.sg](https://data.gov.sg/). To mitigate the impact of missing values, we re-sampled the raw dataset into the 15-minute interval and chose lots with a missing rate of PA of less than 30%. In addition, due to the temporal distribution shift, we only use one-year data (2020/07/01 to 2021/06/30), and the ratio of training: validation: testing sets is set as 10:1:1. We then remove parking lots with obvious distribution shift (i.e., high KL divergence). After sample filtering, it remains 1,687 parking lots with stationary data distributions. We also crawl external attributes for these lots, including meteorological data (i.e., temperature, humidity, and wind speed), panning areas, utilization type, and road networks data from [Data.gov.sg](https://data.gov.sg/), the [Urban Redevelopment Authority (URA)](https://www.ura.gov.sg/) and the [Land Transport Authority (LTA)](https://datamall.lta.gov.sg/content/datamall/en.html) respectively. A detailed description of the dataset can be found in the following table.


  <table>
  <capital></capital>
  <tr>
  <th>Dimension</th>
  <th>Type</th>
  <th>Category</th>
  <th>Feature name</th>
  <th>Detail</th>
  </tr>
  <tr>
  <td >0</td>
  <td >Predict Target</td>
  <td >Parking Availability</td>
  <td >Parking Availability</td>
  <td >Real value</td>
  </tr>
  <tr>
  <td >1</td>
  <td rowspan=6>Temporal Factor<br></td>
  <td rowspan=3>Time-related<br></td>
  <td >Time of day</td>
  <td >0 to 95 int number (24*4)</td>
  </tr>
  <tr>
  <td >2</td>
  <td >Weekday</td>
  <td >0 to 6 int number (7)</td>
  </tr>
  <tr>
  <td >3</td>
  <td >Is_holiday</td>
  <td >One-hot</td>
  </tr>
  <td >4</td>
  <td rowspan=3>Meteorology<br></td>
  <td >Temperature</td>
  <td >Normalized value</td>
  </tr>
  <td >5</td>
  <td >Humidity</td>
  <td >Normalized value</td>
  </tr>
  </tr>
  <td >6</td>
  <td >Windspeed</td>
  <td >Normalized value</td>
  </tr>
  <td >7</td>
  <td rowspan=5>Spatial Factor<br></td>
  <td >Utilization Type</td>
  <td >Utilization Type</td>
  <td >0 to 9 int number (10)</td>
  </tr>
  <td >8</td>
  <td >Region-related</td>
  <td >Planning area</td>
  <td >0 to 35 int number (36)</td>
  </tr>
  </tr>
  <td >9</td>
  <td >Road-related</td>
  <td >Road Density</td>
  <td >Normalized value</td>
  </tr>
  <td >10</td>
  <td rowspan=2>Location<br></td>
  <td >Latitude</td>
  <td >Normalized value</td>
  </tr>
  </tr>
  <td >11</td>
  <td >Longitude</td>
  <td >Normalized value</td>
  </tr>
  </table>
  
  Note: _Normalized_ refers to Z-score normalization, which is applied for fast convergence.

- **Auxiliary Data**. If you would like to visualize the parking lots or customize the adjacency matrix, you can access the parking lot locations in the file `aux_data/lots_location.csv`.

## Requirements
DeepPA uses the following dependencies:
1. Pytorch 1.10 and its dependencies
2. Numpy and Scipy
3. CUDA 11.3 or latest version, cuDNN.

## Folder Structure
We list the code of the major modules as follows:

1. The main function to train/test our model:  [click here.](experiments/DeepPA/main.py "1")
2. The source code of our model: [click here.](src/models/DeepPA.py "2")
3. The trainer/tester: [click here.](src/trainers/deeppa_trainer.py "3")
4. Data preparation and preprocessing are located at [click here.](experiments/DeepPA/main.py "4")
5. Computations: [click here.](src/utils "5")

## Arguments
We introduce some major arguments of our main function here.

Training settings:
- mode: indicating the mode (train or test).
- n_exp: experimental group number.
- gpu: which gpu used to train.
- seed: the random seed for experiments. (default: 0)
- dataset: dataset path for the experiment.
- batch_size: batch size of training or testing.
- seq_len: the length of historical steps.
- horizon: the length of future steps.
- input_dim: the dimension of inputs.
- output_dim: the dimension of inputs.
- max_epochs: maximum number of training epochs.
- patience: the patience of early stopping.
- save_preds: whether to save prediction results.
- wandb: whether to use wandb.

Model hyperparameters:
- dropout: dropout rate.
- n_blocks: number of layers of SLBlock and TLBlock.
- n_hidden: hidden dimensions in SLBlock and TLBlock.
- n_heads: number of heads in MSA.
- spatial_flag: whether to use SLBlock.
- temporal_flag: whether to use TLBlock.
- spatial_encoding: whether to treat temporal factor as a station.
- temporal_encoding: Whether to incorporate spatial factor into TLBlock.
- temporal_PE: whether to use temporal position encoding.
- GCO: whether to use GCO.
- GCO_Thre: the proportion of low frequency signals.
- base_lr: base learning rate.
- lr_decay_ratio: learning rate decay ratio.

## Model training
The following examples are conducted on the base dataset of SINPA:

* Example 1 (DeepPA with default setting):
```
python ./experiments/DeepPA/main.py --dataset /base/ --mode train --gpu 0
```

* Example 2 (DeepPA without GCO):
```
python ./experiments/DeepPA/main.py --dataset /base/ --mode train --gpu 0 --GCO False
```

* Example 2 (DeepPA with the 0.7 proportion of low frequency signals):
```
python ./experiments/DeepPA/main.py --dataset /base/ --mode train --gpu 0 --GCO_Thre 0.7
```

## Model Evaluation
To test the above-trained models, you can use the following command:

* Example 1 (DeepPA with default setting):
```
python ./experiments/DeepPA/main.py --dataset /base/ --mode test --gpu 0
```

* Example 2 (DeepPA with the 0.7 proportion of low frequency signals):
```
python ./experiments/DeepPA/main.py --dataset /base/ --mode test --gpu 0 --GCO_Thre 0.7
```

## License
The <b>SINPA</b> dataset is released under the Singapore Open Data Licence: [https://beta.data.gov.sg/open-data-license](https://beta.data.gov.sg/open-data-license).

## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{zhang2024predicting,
  title={Predicting Parking Availability in Singapore with Cross-Domain Data: A New Dataset and A Data-Driven Approach},
  author={Zhang, Huaiwu and Xia, Yutong and Zhong, Siru and Wang, Kun and Tong, Zekun and Wen, Qingsong and Zimmermann, Roger and Liang, Yuxuan},
  booktitle={Proceedings of the Thirty-third International Joint Conference on Artificial Intelligence, IJCAI-24},
  year={2024}
}
```
