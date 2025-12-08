from celery import Celery
from celery.bin import worker

# Initialize the Celery app
#app = Celery('tasks.content_generation.app')
from tasks.content_generation import app

# Create a worker instance
#worker_instance = worker.worker(app=app)

# Set up arguments for the worker
worker_options = {
    'loglevel': 'info',
    'traceback': True,
    'events': True,    
}

if __name__ == '__main__':
    #app.worker_main(['worker', '--loglevel=info', '--concurrency=4', '-E'])
    app.worker_main(['worker', '--loglevel=info', '-E'])