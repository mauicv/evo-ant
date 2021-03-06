import os
import numpy as np
import time
import sys
import pybullet_envs  # noqa
import gym
import datetime

from gerel.algorithms.RES.population import RESPopulation
from gerel.algorithms.RES.mutator import RESMutator
from gerel.populations.genome_seeders import curry_genome_seeder
from gerel.genome.factories import dense, from_genes
from gerel.util.datastore import DataStore
from gerel.model.model import Model
from batch import BatchJob
from stream_redirect import RedirectAllOutput


ENV_NAME = 'AntBulletEnv-v0'
EPISODES = 5000
STATE_DIMS = 28
ACTION_DIMS = 8
MIN_ACTION = -1
MAX_ACTION = 1
STEPS = 500
LAYER_DIMS = [20, 20]
DIR = 'ant_RES_data'
BATCH_SIZE = 25
batch_job = BatchJob()


@batch_job
def compute_fitness(genomes):
    with RedirectAllOutput(sys.stdout, file=os.devnull), \
            RedirectAllOutput(sys.stderr, file=os.devnull):
        envs = [gym.make(ENV_NAME) for _ in range(len(genomes))]
        models = [Model(genome) for genome in genomes]
        action_map = lambda x: np.tanh(np.array(x))  # noqa
        dones = [False for _ in range(len(genomes))]
        states = [np.array(env.reset(), dtype='float32') for env in envs]
        rewards = [0 for _ in range(len(genomes))]
        for _ in range(STEPS):
            for index, (model, env, done, state) in \
                    enumerate(zip(models, envs, dones, states)):
                if done:
                    continue

                action = model(state)
                action = action_map(action)
                next_state, reward, done, _ = env.step(action)
                rewards[index] += reward
                dones[index] = done
                states[index] = next_state

        # Closing envs fixes memory leak:
        for env in envs:
            env.close()
    return rewards


def make_counter_fn(lim=5):
    def counter_fn():
        if not hasattr(counter_fn, 'count'):
            counter_fn.count = 0
        counter_fn.count += 1
        if counter_fn.count == lim:
            counter_fn.count = 0
            return True
        return False
    return counter_fn


def partition(ls, ls_size):
    parition_num = int(len(ls)/ls_size) + 1
    return [ls[i*ls_size:(i + 1)*ls_size] for i in range(parition_num)]


def departition(ls):
    return [item for sublist in ls for item in sublist]


def print_progress(data):
    data_string = f'{data["run_time"]}: gen: {data["generation"]} '
    for val in ['best_fitness', 'worst_fitness',
                'mean_fitness']:
        data_string += f' {val}: {round(data[val])}'
    data_string += f" ep_time: {data['time']}"
    print(data_string)


if __name__ == '__main__':
    ds = DataStore(name=f'{DIR}')
    generation_inds = [int(i) for i in os.listdir(f'./{DIR}')]
    if generation_inds:
        last_gen_ind = max(generation_inds)
        last_gen = ds.load(last_gen_ind)
        nodes, edges = last_gen['best_genome']
        input_num = len([n for n in nodes if n[4] == 'input'])
        output_num = len([n for n in nodes if n[4] == 'output'])
        nodes = [n for n in nodes if n[4] == 'hidden']
        genome = from_genes(
            nodes, edges,
            input_size=input_num,
            output_size=output_num,
            weight_low=-2,
            weight_high=2,
            depth=len(LAYER_DIMS))
        next_gen = last_gen_ind + 1
        print(f'seeding generation {next_gen} with last best genome: {genome}')
    else:
        genome = dense(
            input_size=STATE_DIMS,
            output_size=ACTION_DIMS,
            layer_dims=LAYER_DIMS
        )
        print(f'seeding generation 0, with  genome: {genome}')

    weights_len = len(genome.edges) + len(genome.nodes)
    init_mu = np.array(genome.weights)

    mutator = RESMutator(
        initial_mu=init_mu,
        std_dev=0.5,
        alpha=1
    )

    seeder = curry_genome_seeder(
        mutator=mutator,
        seed_genomes=[genome]
    )

    population = RESPopulation(
        population_size=250,
        genome_seeder=seeder
    )

    print('population size = 250')
    print('population type = RESPopulation')
    print('mutator type = RESMutator')
    print('num episodes:', EPISODES)
    print('num steps:', STEPS)
    print(f'running algorithm on: {batch_job.num_processes} CPUs')
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print("Current Time:", current_time)

    init_time = time.time()
    for episode in range(EPISODES):
        start = time.time()
        genes = [g.to_reduced_repr for g in population.genomes]
        partitioned_population = partition(genes, BATCH_SIZE)
        scores = departition(compute_fitness(partitioned_population))
        for genome, fitness in zip(population.genomes, scores):
            genome.fitness = fitness
        data = population.to_dict()
        mutator(population)
        ds.save(data)
        end = time.time()
        episode_time = f'{round(end - start)} secs'
        current_run_time = str(
            datetime.timedelta(seconds=round(end - init_time)))
        print_progress({
            **data,
            'time': episode_time,
            'run_time': current_run_time
        })
