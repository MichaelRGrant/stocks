#!/bin/bash

sudo apt-get install tmux

wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh -b -p $HOME/anaconda3
export PATH="$HOME/anaconda3/bin:$PATH"

conda create -n stocks -y

conda install jupyter notebook ipykernel -y
conda install -c conda-forge nb_black -y

pip install pandas==1.0 pyarrow==1.0.1 scikit-learn==0.23.2 scikit-optimize tensorflow numpy matplotlib seaborn tqdm ta-lib easydict jedi==0.17.2

current_env=`echo $CONDA_DEFAULT_ENV`

python -m ipykernel install --user --name=$current_env
