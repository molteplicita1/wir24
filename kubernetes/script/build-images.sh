echo "Clearing minikube images"
minikube image list | grep rag- | xargs minikube image rm

docker images --format "{{.Repository}}:{{.Tag}}" | grep "chromadb/chroma" | xargs docker rmi
docker image pull chromadb/chroma:0.5.15
echo "Pushing chroma to minikube"
minikube image load chromadb/chroma:0.5.15


cd gateway
docker images --format "{{.Repository}}:{{.Tag}}" | grep "rag-gateway" | xargs docker rmi
echo "Starting gateway build"
docker build --no-cache -t rag-gateway:latest .
echo "Pushing gateway to minikube"
minikube image load rag-gateway:latest

                    
cd ../retrieve
docker images --format "{{.Repository}}:{{.Tag}}" | grep "rag-retrieve" | xargs docker rmi
echo "Starting retrieve build"
docker build --no-cache -t rag-retrieve:latest .
echo "Pushing retrieve to minikube"
minikube image load rag-retrieve:latest

    
cd ../populate
docker images --format "{{.Repository}}:{{.Tag}}" | grep "rag-populate" | xargs docker rmi
echo "Starting populate build"
docker build --no-cache -t rag-populate:latest .
echo "Push populate to minikube"
minikube image load rag-populate:latest

      
cd ../generate
docker images --format "{{.Repository}}:{{.Tag}}" | grep "rag-generate" | xargs docker rmi
echo "Starting generate build"
docker build --no-cache -t rag-generate:latest .
echo "Push generate to minikube"
minikube image load rag-generate:latest


minikube image list | grep rag-
minikube image list | grep chroma