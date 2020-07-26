import time

def get_time_str(delta_t):
    time_units = ['s', 'ms', 'us']
    k = 0
    for i in range(len(time_units)):
        k = i
        if delta_t > 1:
            break
        delta_t *= 1000
    return f"{delta_t:.2f} {time_units[k]}"


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        print(f"{method.__name__} time : {get_time_str(time.time() - ts)}")
        return result
    return timed
