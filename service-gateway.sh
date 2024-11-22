#!/bin/bash
kubectl delete deployment generate 
kubectl delete deployment populate
kubectl delete deployment retrieve
kubectl delete deployment gateway
./build-images.sh
./kube.sh
sleep 15
minikube service gateway --url