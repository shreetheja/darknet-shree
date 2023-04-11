# Ubuntu 18.04 Software
FROM ubuntu:18.04

# Copy the current Repo to directory
COPY . /app

# Change directory to App
WORKDIR /app

# Update the Packages
RUN apt update

# Install Make For Installing Dependencies
RUN apt install make -y

# Install Wget to Get the Weights
RUN apt install wget -y 

# Gcc is needed to Run C Scripts
RUN apt install gcc -y

# G++ is needed to Run C++ Scripts
RUN apt install g++ -y

# INstall Dependencies
RUN make

# Download Weights
RUN wget https://pjreddie.com/media/files/yolov3.weights