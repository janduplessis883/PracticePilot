import time
from loguru import logger

# Add a file sink for logging
logger.add("../../logs/app.log", rotation="10 MB", retention="28 days")


# = Decorators =========================================================================================================

def time_it(func):
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"🖥️    Started: '{func_name}'")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        func_name = func.__name__
        logger.info(f"✅ Completed: '{func_name}' ⚡️{elapsed_time:.6f} sec")
        return result

    return wrapper


# Universal functions ==================================================================================================
