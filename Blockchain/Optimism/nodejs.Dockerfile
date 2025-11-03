# Use official Node.js v20.10.0 image
FROM node:20.10.0

# Set working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Install app dependencies
COPY package.json /usr/src/app/
RUN npm install

COPY . /usr/src/app
