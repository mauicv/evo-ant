import matplotlib.pyplot as plt
from gerel.util.datastore import DataStore


if __name__ == '__main__':
    ds = DataStore(name='ant_RES_data')
    data = list(ds.generations())
    mean = [generation['mean_fitness'] for generation in data]
    best = [generation['best_fitness'] for generation in data]
    worst = [generation['worst_fitness'] for generation in data]
    plt.plot(mean)
    plt.plot(best)
    plt.plot(worst)
    plt.show()
