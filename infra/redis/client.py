import redis

client = redis.from_url('redis://localhost:6379/0')

try:
    response = client.ping()
    print(f"Соединение установлено: {response}")  # True
except redis.ConnectionError:
    print("Ошибка подключения к Redis")