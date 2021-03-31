from batch import BatchJob
import time
import pybullet_envs  # noqa
import gym
import numpy as np


STEPS = 100
ENV_NAME = 'CartPole-v0'

batch_job = BatchJob(num_processes=8)


def get_open_fds():
    '''
    return the number of open file descriptors for current process

    .. warning: will only work on UNIX-like os-es.
    '''
    import subprocess
    import os

    pid = os.getpid()
    procs = subprocess.check_output(
        ["lsof", '-w', '-Ff', "-p", str(pid)])
    procs = procs.decode('ascii')
    nprocs = len(
        list(filter(
            lambda s: s and s[0] == 'f' and s[1:].isdigit(),
            procs.split('\n'))
        ))
    return nprocs


def compute_fitness(genomes):
    envs = [gym.make(ENV_NAME) for _ in range(len(genomes))]
    dones = [False for _ in range(len(genomes))]
    states = [np.array(env.reset(), dtype='float32') for env in envs]
    rewards = [0 for _ in range(len(genomes))]

    for _ in range(STEPS):
        for index, (env, done, state) in \
                enumerate(zip(envs, dones, states)):
            if done:
                continue

            next_state, reward, done, _ = \
                env.step(env.action_space.sample())
            rewards[index] += reward
            dones[index] = done
            states[index] = next_state

    # Closing envs fixes memory leak:
    for env in envs:
        env.close()

    return rewards


if __name__ == '__main__':
    compute_fitness_batch = batch_job(compute_fitness)

    print('comparing parrallel compute times and checking for memory leaks')

    print('num open files:', get_open_fds())
    start = time.time()
    for i in range(100):
        compute_fitness_batch([[0 for i in range(100)] for _ in range(1)])
    end = time.time()
    print('time (100, 1): ', end - start)
    print('num open files:', get_open_fds())
    start = time.time()
    for i in range(100):
        compute_fitness_batch([[0 for i in range(50)] for _ in range(2)])
    end = time.time()
    print('time: (50, 2) ', end - start)
    print('num open files:', get_open_fds())
    start = time.time()
    for i in range(100):
        compute_fitness_batch([[0 for i in range(25)] for _ in range(4)])
    end = time.time()
    print('time: (25, 4)', end - start)
    print('num open files:', get_open_fds())
    start = time.time()
    for i in range(100):
        compute_fitness_batch([[0 for i in range(20)] for _ in range(5)])
    end = time.time()
    print('time: (20, 5)', end - start)
    print('num open files:', get_open_fds())
    start = time.time()
    for i in range(100):
        compute_fitness_batch([[0 for i in range(10)] for _ in range(10)])
    end = time.time()
    print('time: (10, 10)', end - start)
    print('num open files:', get_open_fds())
    start = time.time()
    for i in range(100):
        compute_fitness_batch([[0 for i in range(5)] for _ in range(20)])
    end = time.time()
    print('time: (5, 20)', end - start)
    print('num open files:', get_open_fds())
    start = time.time()
    for i in range(100):
        compute_fitness_batch([[0 for i in range(2)] for _ in range(50)])
    end = time.time()
    print('time: (2, 50)', end - start)
    print('num open files:', get_open_fds())
    start = time.time()
    for i in range(100):
        compute_fitness_batch([[0 for i in range(1)] for _ in range(100)])
    end = time.time()
    print('time: (1, 100)', end - start)
    print('num open files:', get_open_fds())
