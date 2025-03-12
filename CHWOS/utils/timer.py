import time

from CHWOS.utils.log import get_logger

logger = get_logger(__name__)


def get_time_str_from_seconds(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}:{int(minutes)}:{round(seconds)} (h:m:s)"


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"[TIMER] [{func.__name__}] took {get_time_str_from_seconds(end_time - start_time)}")
        return result

    return wrapper
