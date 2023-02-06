# de(ep)tangle GUI app

## Setup
In order to setup the app, one needs to install the necessary dependencies.
We recommend the use of a virtual environment to do so.
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-gui.txt
```

In addition, the de(ep)tangle library also needs to be installed 
```
cd ../
pip install .
```

Likewise, you will need the weights and the video sample if you want to try it out first.
```
wget https://sid.erda.dk/share_redirect/c3n4DHbGdI -O celegans_512.avi
wget https://sid.erda.dk/share_redirect/cEjIpG1yQl -O weights.zip
unzip weights.zip
```

You also need to install `ffmpeg`, e.g. by
```
sudo apt install ffmpeg
```


## Usage
To run, one only needs to write
```
python3 app.py
```

## Screenshot
<p align="center">
  <img src="https://github.com/kirkegaardlab/deeptanglelabel/blob/main/docs/figures/gui.png" height="512" />
</p>
