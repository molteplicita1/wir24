cd ../retrieve
kubectl apply -f kubernetes/
echo "Kubernetes apply retrieve" 

cd ../gateway
kubectl apply -f kubernetes/
echo "Kubernetes apply gateway"   

cd ../populate
kubectl apply -f kubernetes/
echo "Kubernetes apply populate"  

cd ../generate
kubectl apply -f kubernetes/
echo "Kubernetes apply generate"    