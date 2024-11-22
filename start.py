import subprocess


def start_services():
    services = [
        ["uvicorn", "populate.main:app", "--port", "8001"],
        ["uvicorn", "retrieve.main:app", "--port", "8002"],
        ["uvicorn", "generate.main:app", "--port", "8003"],
        ["uvicorn", "gateway.main:app", "--port", "8004"],
    ]
    processes = [subprocess.Popen(service) for service in services]
    for p in processes:
        p.wait()


if __name__ == "__main__":
    start_services()
