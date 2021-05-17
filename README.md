# Sentiment Classification
Sentiment classification of text input into positive / neutral / negative classes.

**Model Architecture Used: BERT**

Reference: Google official documents - https://github.com/tensorflow/docs/tree/master/site/en/tutorials/text


**Dataset used: 3 class twitter sentiment analysis data**
Reference: Kaggle - https://www.kaggle.com/seriousran/appletwittersentimenttexts


## For training:
* Refer `./train/train.ipynb`
* Dependancies are listed in `requirements.txt`
* For inference on custom data see `predict_on_custom_data.ipynb`
* The model file trained using this script is available at model file
* The dataset used for training is from Kaggle - https://www.kaggle.com/seriousran/appletwittersentimenttexts
* The dataset is made available at `./data/apple-twitter-sentiment-texts.csv`
* The pretrained model is used from tensorflow hub

## For Inference:
* Refer `predict_on_custom_data.ipynb` - All the below steps are also mentioned there.
* Dependancies are listed in requirements.txt
* The pre-trained model file is available here - https://drive.google.com/file/d/1zQ0WQ8IugTYGKh5rriCxj_FLusXFuZI4/view?usp=sharing
* for both the below mentioned evaluation/testing methods, you need to download the file, keep the **file as it is with the same name** in the `./model/` directory.

* Both the inference methods are tested on linux machines on CPU with 8GB RAM


### Prediction using the docker

*If you are not familiad with docker, just follow the instructions. Begin with installing docker from - https://docs.docker.com/engine/install/*

1. You should have docker installed for this to work
2. Clone the repo - `git clone https://github.com/renjithbaby23/sentiment-classification.git`
3. cd to the repo - `cd sentiment-classification`
4. Enusre that you have downloaded the [model file](https://drive.google.com/file/d/1zQ0WQ8IugTYGKh5rriCxj_FLusXFuZI4/view?usp=sharing) and is available at `./model/model.tar.gz`
4. Build the docker image - `docker build ./ -t renjith/sentiment:v1`
5. Run the docker container - `docker run -d --name sentiment-serving -p 80:80 renjith/sentiment:v1`
6. From the same machine where you ran the docker, go to the url -`http://localhost:80/sentiment-detection/health` to check if the service is up and running.
7. Run the cells under **Custom prediction - method 1** title of the `predict_on_custom_data.ipynb` notebook


### Preduction using notebook or python script

1. Create a conda environment with python 3.8 `sentiment`
2. Activate the environment `sentiment`
3. Install the requirements `pip install -r requirements.txt`
5. Untar the [model file](https://drive.google.com/file/d/1zQ0WQ8IugTYGKh5rriCxj_FLusXFuZI4/view?usp=sharing) using `tar xzf model.tar.gz`
6. keep the extracted folder **without renaming** - here `./model/apple5_model/`
7. If the TF saved model is kept in a different folder, update the below variable `saved_model_path`
8. Modify the list `test_data` with your custom text to see the predictions
9. Run the cells under **Custom prediction - method 2** title of the `predict_on_custom_data.ipynb` notebook
10. You can also run the evaluation.py once you are done with the steps till 7
