echo "Install pytorch from source, ready for old GPU "

echo "PRESS [ENTER] TO CONTINUE THE INSTALLATION"
echo "IF YOU WANT TO CANCEL, PRESS [CTRL] + [C]"
read

echo "[Set the env]"
sudo apt-get install python3-pip python3-dev python-virtualenv
virtualenv --system-site-packages -p python3 ~/pytorch_s
source ~/pytorch_s/bin/activate


echo "[Download and install pytorch from source]"
pip install pyyaml
pip install numpy
cd
cd Documents
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v0.4.1
git submodule update --init
sudo apt install cmake
python setup.py install

pip install https://files.pythonhosted.org/packages/8c/52/33d739bcc547f22c522def535a8da7e6e5a0f6b98594717f519b5cb1a4e1/torchvision-0.1.8-py2.py3-none-any.whl

echo "[Set the ROS evironment]"
sh -c "echo \"alias sps='source ~/pytorch_s/bin/activate'\" >> ~/.bashrc"

source $HOME/.bashrc


echo "Shortcut setting"
echo "Everytime you want to get into virtualenv of pytorch, just typing [ sps ]"

echo "Install Finish!!!"

