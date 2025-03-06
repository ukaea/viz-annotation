from celery import Celery

app = Celery('tasks',
    broker="redis://redis:6379/0",  # Redis as a message broker
    backend="redis://redis:6379/0"  # Redis as result backend 
)

@app.task
def add(x, y):
    return x + y