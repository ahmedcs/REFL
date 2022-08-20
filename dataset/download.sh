#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # no color
DIR="./data"

# Download and decompress datasets

Help()
{
   # Display Help
   echo "We provide four datasets (open_images, stackoverflow, and speech)"
   echo "to evalute the performance of Kuiper"
   echo 
   echo "Syntax: ./download.sh [-g|h|v|V]"
   echo "options:"
   echo "-h     Print this Help."
   echo "-s     Download Speech Commands dataset (about 2.3GB)"
   echo "-o     Download Open Image dataset (about 66GB)"
   echo "-r     Download Reddit dataset (about 25G)"
   echo "-f     Download Stackoverflow dataset (about 800M)"
   echo "-c     Download CIFAR10 dataset (about 170M)"
}

speech()
{
    if [ ! -d "${DIR}/google_speech/train/" ];
    then
        echo "Downloading Speech Commands dataset(about 2.4GB)..."
        wget -O ${DIR}/google_speech/google_speech.tar.gz https://fedscale.eecs.umich.edu/dataset/google_speech.tar.gz

        echo "Dataset downloaded, now decompressing..." 
        tar -xf ${DIR}/google_speech/google_speech.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/google_speech/google_speech.tar.gz

        echo -e "${GREEN}Speech Commands dataset downloaded!${NC}"
    else
        echo -e "${RED}Speech Commands dataset already exists under ${DIR}/google_speech/!"
fi
}

openimage()
{
    if [ ! -d "${DIR}/openimage/train/" ];
    then
        echo "Downloading Open Images dataset(about 66GB)..."   
        wget -O ${DIR}/openimage/openImage.tar.gz https://fedscale.eecs.umich.edu/dataset/openImage.tar.gz
        
        echo "Dataset downloaded, now decompressing..." 
        tar -xf ${DIR}/openimage/openImage.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/openimage/openImage.tar.gz

        echo -e "${GREEN}Open Images dataset downloaded!${NC}"
    else
        echo -e "${RED}Open Images dataset already exists under ${DIR}/openimage/!"
fi
}

reddit() 
{
    if [ ! -d "${DIR}/reddit/train/" ]; 
    then
        echo "Downloading reddit dataset(about 25G)..."   
        wget -O ${DIR}/reddit/reddit.tar.gz https://fedscale.eecs.umich.edu/dataset/reddit.tar.gz
        
        echo "Dataset downloaded, now decompressing..." 
        tar -xf ${DIR}/reddit/reddit.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/reddit/reddit.tar.gz

        echo -e "${GREEN}reddit dataset downloaded!${NC}"
    else
        echo -e "${RED}reddit dataset already exists under ${DIR}/reddit/!"
fi
}


stackoverflow()
{
    if [ ! -d "${DIR}/stackoverflow/train/" ];
    then
        echo "Downloading stackoverflow dataset(about 800M)..."
        wget -O ${DIR}/stackoverflow/stackoverflow.tar.gz https://fedscale.eecs.umich.edu/dataset/stackoverflow.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/stackoverflow/stackoverflow.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/stackoverflow/stackoverflow.tar.gz

        echo -e "${GREEN}stackoverflow dataset downloaded!${NC}"
    else
        echo -e "${RED}stackoverflow dataset already exists under ${DIR}/stackoverflow/!"
fi
}

cifar10()
{
    if [ ! -d "${DIR}/cifar10/cifar-10-batches-py/" ];
    then
        echo "Downloading cifar10 dataset(about 170M)..."
        wget -O ${DIR}/cifar10/cifar10.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

        echo "Dataset downloaded, now decompressing..."
        tar -xf ${DIR}/cifar10/cifar10.tar.gz -C ${DIR}

        echo "Removing compressed file..."
        rm -f ${DIR}/cifar10/cifar10.tar.gz

        echo -e "${GREEN}CIFAR10 dataset downloaded!${NC}"
    else
        echo -e "${RED}CIFAR10 dataset already exists under ${DIR}/cifar-10-batches-py/!"
fi
}


while getopts ":hsoacegildrtw" option; do
   case $option in
      h ) # display Help
         Help
         exit;;
      o )
         open_images   
         ;;    
      s )
         speech   
         ;;
      r )
         reddit
         ;;
      f )
         stackoverflow
         ;;
      c )
         cifar10
         ;;         
      \? ) 
         echo -e "${RED}Usage: cmd [-h] [-A] [-o] [-t] [-p]${NC}"
         exit 1;;
   esac
done

if [ $OPTIND -eq 1 ]; then 
    echo -e "${RED}Usage: cmd [-h] [-A] [-o] [-t] [-p]${NC}"; 
fi
