# Train-models

## Run script
Python >= 3.10.x and Linux is required.
```bash=
cd NetGuardia-ML/src
chmod +x main.py
./main.py -s [2017 or 2018] -a
```

## Run docker
```
cd NetGuardia-ML/src
docker pull ghcr.io/parrotxray/netguardia-train:latest
docker run --gpus all \
  -v ./rawdata:/app/rawdata \
  -v ./outputs:/app/outputs \
  -v ./artifacts:/app/artifacts \
  -v ./metadata:/app/metadata \
  -v ./plots:/app/plots \
  netguardia-train ./main.py -s [2017 or 2018] -a
```
