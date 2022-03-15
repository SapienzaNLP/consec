#!/bin/bash

# setup conda
source ~/miniconda3/etc/profile.d/conda.sh

# create conda env
read -rp "Enter environment name: " env_name
read -rp "Enter python version (e.g. 3.7): " python_version
conda create -yn "$env_name" python="$python_version"
conda activate "$env_name"

# install torch
read -rp "Enter cuda version (e.g. '10.1' or 'none' to avoid installing cuda support): " cuda_version
if [ "$cuda_version" == "none" ]; then
    conda install -y pytorch torchvision cpuonly -c pytorch
else
    conda install -y pytorch torchvision cudatoolkit=$cuda_version -c pytorch
fi

# install python requirements
pip install -r requirements.txt

# download nltk wordnet
python -c "import nltk; nltk.download('wordnet')"

# install java (needed to run raganato eval)
sudo apt-get update && sudo apt-get install -y openjdk-11-jdk

# install xmllint (needed for xml beautification)
sudo apt-get install libxml2-utils

# download raganato framework
read -p "Download raganato framework? [y/N] "
if [[ $REPLY =~ ^[Yy]$ ]]
then

  wget -P data/ http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
  unzip -d data/ data/WSD_Evaluation_Framework.zip
  rm data/WSD_Evaluation_Framework.zip

  # download WNG and WNE
  read -p "Download WNG and WNE? [y/N] "
  if [[ $REPLY =~ ^[Yy]$ ]]
  then
    mkdir data/WSD_Evaluation_Framework/Training_Corpora/WNG/
    wget -P data/WSD_Evaluation_Framework/Training_Corpora/WNG/ https://raw.githubusercontent.com/SapienzaNLP/ewiser/master/res/corpora/training/orig/glosses_main.data.xml
    wget -P data/WSD_Evaluation_Framework/Training_Corpora/WNG/ https://raw.githubusercontent.com/SapienzaNLP/ewiser/master/res/corpora/training/orig/glosses_main.gold.key.txt

    mkdir data/WSD_Evaluation_Framework/Training_Corpora/WNE/
    wget -P data/WSD_Evaluation_Framework/Training_Corpora/WNE/ https://raw.githubusercontent.com/SapienzaNLP/ewiser/master/res/corpora/training/orig/examples.data.xml
    wget -P data/WSD_Evaluation_Framework/Training_Corpora/WNE/ https://raw.githubusercontent.com/SapienzaNLP/ewiser/master/res/corpora/training/orig/examples.gold.key.txt
  fi

fi
