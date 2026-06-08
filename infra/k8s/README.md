# JMLC minikube infrastructure

Рабочий dev-mode набор:

- `redis` в minikube для LangGraph checkpointing;
- `gemma-host` как Kubernetes service на внешний GPU Docker endpoint;
- сама Gemma запускается вне minikube через обычный Docker с `--gpus all`.

## Minikube

```powershell
minikube start --driver=docker --apiserver-name=localhost
kubectl apply -f .\infra\k8s\namespace.yaml
kubectl apply -f .\infra\k8s\redis.yaml
kubectl apply -f .\infra\k8s\neo4j.yaml
kubectl apply -f .\infra\k8s\gemma-external.yaml
```

Проверка:

```powershell
kubectl -n jmlc get pods,svc,pvc
```

## Neo4j

Neo4j runs inside the `jmlc` namespace as service `neo4j`.

Internal endpoints for services in the cluster:

```text
bolt://neo4j:7687
http://neo4j:7474
```

Local port-forward for integration tests from the host:

```powershell
kubectl -n jmlc port-forward svc/neo4j 7687:7687 7474:7474
```

Default dev credentials:

```text
username: neo4j
password: password
```

## External Gemma

Если образа еще нет, его можно собрать:

```powershell
docker build -t graphmemory-gemma:dev .\infra\gemma
```

Запуск Gemma на хосте:

```powershell
docker run -d --name jmlc-gemma-host --gpus all `
  -p 8000:8000 `
  -e HF_TOKEN=$env:HF_TOKEN `
  -v "${PWD}\hf-cache:/models/huggingface:ro" `
  graphmemory-gemma:dev `
  /models/huggingface/google/gemma-3-4b-it `
  --host 0.0.0.0 `
  --port 8000 `
  --served-model-name gemma-3-4b-it
```

Проверка с хоста:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/models
```

Проверка из minikube:

```powershell
kubectl -n jmlc run curl-gemma-host --rm -i --restart=Never --image=curlimages/curl -- `
  curl -sS http://gemma-host:8000/v1/models
```

Внутренний endpoint для сервисов в кластере:

```text
http://gemma-host:8000/v1
```
