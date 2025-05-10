#!/bin/bash

# Start Minikube if not running
minikube status || minikube start

# Set docker env to use Minikube's Docker daemon
eval $(minikube docker-env)

# Build Docker images
docker build -t weather-predictor:latest ../

# Apply Kubernetes manifests
kubectl apply -f weather-deployment.yml
kubectl apply -f weather-service.yml

# Wait for pods to be ready
echo "Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=weather-predictor --timeout=120s

# Get NodePort URL
echo "Weather Predictor can be accessed at:"
minikube service weather-predictor --url