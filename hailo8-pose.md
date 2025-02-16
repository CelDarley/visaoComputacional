
dpkg -i hnote: This is an issue with the package mentioned above, not pip.  
source venv_hailo_rpi5_examples/bin/activate  
pip install -r requirements.txt  
pip install hailo    
unzip /home/darley/Downloads/tappas_3.31.0_linux_installer.zip   
cd tappas_v3.31.0/  
./install.sh   
source setup_env.sh  
echo $HAILO_DRIVER_PATH    
echo $PKG_CONFIG_PATH  
hailortcli fw-control identify  
cd basic_pipelines/  
./download_resources.sh  
sudo apt update  
sudo apt full-upgrade  
sudo apt install hailo-all  
reboot  
source venv_hailo_rpi5_examples/bin/activate  
cd hailo-rpi5-examples/  
sudo apt install --reinstall hailo-all  
reboot  
root@raspberrypi:/home/darley/hailo-rpi5-examples/basic_pipelines#    

sudo su  
cd hailo-rpi5-examples/  
source setup_env.sh  
cd basic_pipelines/  
python3 pose_estimation.py   

