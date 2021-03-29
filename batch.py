"""
Simple wrapper to compute a parrallelize a function.

given a function f that takes a single argument, `arg`, the batch_job decorator
turns f into a function that takes a list of args and computes each result in
a seperate process. It then returns the results as a list.

___

use:
    ```
    batch_job = BatchJob(num_processes=3)

    @batch_job
    def job(arg):
        # do something...

    job([args_1, args_2, ....])
    ```

The above runs: job(args_1), job(args_2), ... in parrallel and returns
[job(args_1), job(args_2), ...]
"""


from multiprocessing import Process, Queue, cpu_count

PROCESSES = cpu_count() - 1


class BatchJob:
    def __init__(self, num_processes=PROCESSES):
        self.in_queue = Queue()
        self.out_queue = Queue()
        self.processes = []
        self.num_processes = num_processes

    def __call__(self, job):
        def batched_job(arg_batch):
            for ind, arg in enumerate(arg_batch):
                self.in_queue.put((ind, arg))

            def process_task(task_queue, return_queue):
                while not task_queue.empty():
                    args_ind, args = task_queue.get()
                    result = job(args)
                    return_queue.put((args_ind, result))
                return True

            for i in range(self.num_processes):
                p = Process(target=process_task,
                            args=(self.in_queue, self.out_queue))
                p.start()
                self.processes.append(p)

            for p in self.processes:
                p.join()

            results = []
            while not self.out_queue.empty():
                results.append(self.out_queue.get())

            results.sort(key=lambda item: item[0])
            return [r for i, r in results]
        return batched_job
