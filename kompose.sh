cd gateway
kubectl apply -f kubernetes/
echo "Kubernetes apply gateway"

cd ../generate
kubectl apply -f kubernetes/
echo "Kubernetes apply generate" 

cd ../populate
kubectl apply -f kubernetes/
echo "Kubernetes apply populate"

cd ../retrieve
kubectl apply -f kubernetes/
echo "Kubernetes apply retrieve" 