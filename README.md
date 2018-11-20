# pytorch_learning

## install env

> sudo apt-get install python3-pip python3-dev python-virtualenv

> virtualenv --system-site-packages -p python3 ~/pytorch

> source ~/pytorch/bin/activate

# setup remote jypternotebook server

> jupyter notebook --generate-config

> cd ~/.jupyter/

Find "NotebookApp.allow_remote_access : False" and set it Teue

> ./ngrok http 8888
