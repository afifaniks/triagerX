## Create Conda Environment
```shell
conda create --name trx --file requirements.txt
```
## Run Jupyter NB Server on ARC Cluster
```shell
sbatch ./start_nb_server_job.sh
```

## Stop Server
```shell
./stop_nb_server_job.sh
```
