# Agnostic Federated Learning

### How to run
```
python main.py 
```
options:
```
  --dataset {mnist,cifar10}          
  --federated_type {fedavg,afl}     
  --model {cnn,mlp}         
  --n_clients int            
  --global_epochs int    
  --local_epochs int
  --batch_size int
  --on_cuda {yes,no}
  --optimizer {sgd,adam}
  --lr float
  --iid {yes,no}
  --drfa_gamma float
```

### Run experiment

#### IID

##### cifar10
```
# FedAvg
python -u main.py --dataset cifar10 --from_csv iid--federated_type fedavg >> log/cifar10avg.log
# AFL
python -u main.py --dataset cifar10 --from_csv iid --federated_type afl  >> log/cifar10afl.log
```
##### mnist
```
# FedAvg
python -u main.py --dataset mnist --from_csv iid--federated_type fedavg >> log/mnistavg.log
# AFL
python -u main.py --dataset mnist --from_csv iid --federated_type afl  >> log/mnistafl.log
```

##### fashionmnist
```
# FedAvg
python -u main.py --dataset fashionmnist --from_csv iid--federated_type fedavg >> log/fashionmnistavg.log
# AFL
python -u main.py --dataset fashionmnist --from_csv iid --federated_type afl  >> log/fashionmnistafl.log
```
