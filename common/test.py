from concurrent.futures import ThreadPoolExecutor
from time import process_time

def processor_intensive(arg):
    def fib(n):
        return fib(n - 1) + fib(n - 2) if n > 2 else n
    
    start_time = process_time()
    result = fib(arg)
    end_time = process_time()
    return end_time - start_time, result

def manager(calc_stuff):
    inputs = range(25, 32)
    timings, results = [], []
    start = process_time()

    with ThreadPoolExecutor() as executor:
        for timing, result in executor.map(calc_stuff, inputs):
            timings.append(timing)
            results.append(result)

        finish = process_time()

    print(f'{calc_stuff.__name__}')
    print(f'wall time to execute: {finish-start}')
    print(f'total of timings for each call: {sum(timings)}')
    print(f'time saved by parallelizing: {sum(timings) - (finish-start)}')
    print(dict(zip(inputs, results)), end = '\n\n')


if __name__ == "__main__":
    manager(processor_intensive)

