import os
import numpy as np
from gerel.algorithms.RES.population import RESPopulation
from gerel.algorithms.RES.mutator import RESMutator
from gerel.populations.genome_seeders import curry_genome_seeder
from gerel.genome.factories import dense, from_genes
from gerel.util.datastore import DataStore
from gerel.model.model import Model
from batch import BatchJob
import time
from stream_redirect import RedirectStream
import sys
import pybullet_envs  # noqa
import gym


ENV_NAME = 'AntBulletEnv-v0'
EPISODES = 500
STATE_DIMS = 28
ACTION_DIMS = 8
MIN_ACTION = -1
MAX_ACTION = 1
STEPS = 500
LAYER_DIMS = [20, 20]
batch_job = BatchJob()


@batch_job
def compute_fitness(genomes):
    with RedirectStream(sys.stdout), RedirectStream(sys.stderr):
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
    data_string = ''
    for val in ['generation', 'best_fitness', 'worst_fitness', 'mean_fitness']:
        data_string += f' {val}: {data[val]}'
    print(data_string)


if __name__ == '__main__':
    ds = DataStore(name='ant_RES_data')
    generation_inds = [int(i) for i in os.listdir('./ant_RES_data')]
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
    else:
        genome = dense(
            input_size=STATE_DIMS,
            output_size=ACTION_DIMS,
            layer_dims=LAYER_DIMS
        )

    next_gen = last_gen_ind + 1
    print(f'seeding generation {next_gen} with last best genome: {genome}')
    weights_len = len(genome.edges) + len(genome.nodes)
    init_mu = np.random.uniform(-1, 1, weights_len)

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

    for episode in range(EPISODES):
        start = time.time()
        genes = [g.to_reduced_repr for g in population.genomes]
        partitioned_population = partition(genes, 25)
        scores = departition(compute_fitness(partitioned_population))
        for genome, fitness in zip(population.genomes, scores):
            genome.fitness = fitness
        data = population.to_dict()
        mutator(population)

        print_progress(data)
        ds.save(data)
        end = time.time()
        print(f'time: {end - start}')
