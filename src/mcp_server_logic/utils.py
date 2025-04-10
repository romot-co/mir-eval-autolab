import uuid
import time

def generate_id():
    return str(uuid.uuid4())

def get_timestamp():
    return time.time() 