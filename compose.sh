docker compose down
docker compose down --rmi local

docker images --format "{{.Repository}}:{{.Tag}}" | grep "rag-gateway" | xargs docker rmi
docker images --format "{{.Repository}}:{{.Tag}}" | grep "rag-generate" | xargs docker rmi
docker images --format "{{.Repository}}:{{.Tag}}" | grep "rag-populate" | xargs docker rmi
docker images --format "{{.Repository}}:{{.Tag}}" | grep "rag-retrieve" | xargs docker rmi
docker images --format "{{.Repository}}:{{.Tag}}" | grep "chromadb/chroma" | xargs docker rmi

docker compose up --build -d

docker compose logs -f