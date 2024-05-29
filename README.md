# Triager X
## Build docker image
```shell

docker build -t triagerx .
```
## Run docker container
```shell

docker run --gpus all --rm -p 8000:80 --name triagerx triagerx
```