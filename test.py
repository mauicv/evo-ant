from batch import BatchJob
import time
import pybullet_envs  # noqa
import gym
import numpy as np


STEPS = 100
ENV_NAME = 'CartPole-v0'

batch_job = BatchJob()


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

    print(f'Computing best batch_size for {batch_job.num_processes} CPUs')

    batch_data = [(100, 1), (50, 2), (25, 4), (20, 5), (10, 10), (5, 20),
                  (2, 50), (1, 100)]

    for batch_size, batches in batch_data:

        start = time.time()
        for i in range(100):
            compute_fitness_batch([[0 for i in range(100)] for _ in range(1)])
        end = time.time()
        s = f'BATCH_SIZE={batch_size}, BATCHES={batches}, time: {end - start}'
        print(s)
