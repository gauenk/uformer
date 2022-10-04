#!/bin/bash
CWD=$(pwd)
python -m pip install pip --upgrade
cd ../
git clone git@github.com:gauenk/data_hub.git
git clone git@github.com:gauenk/cache_io.git
git clone git@github.com:gauenk/dnls.git
cd ./data_hub
python -m pip install -e .
cd ../cache_io
python -m pip install -e .
cd ../dnls
python -m pip install -e .
cd ../uformer
python -m pip install -e .
