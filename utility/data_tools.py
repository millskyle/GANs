import numpy as np

class DataStreamer(object):
    """
    Class for reading csv data from the output file of generateconfigs.jl,
    and providing shuffled batching.
    """
    def __init__(self, csvfile, fake=False):
        with open(csvfile, 'rb') as F:
            cleaned_lines = (line.replace(b';',b',') for line in F)
            data  = np.genfromtxt(cleaned_lines, dtype=int, delimiter=',')
            N = int(((data.shape[1] - 1) / 2))
            configs = data[:,-(N+1):-1].reshape((-1,N))

        self.data = configs
        self.data[self.data==0]=-1
        self.example_shape = self.data.shape[1:]
        self.epoch = -1
        self.__new_epoch()
        logging.info("Data shape: " + str(self.data.shape))

        """Here I (conditionally) override the csv data with some 'easy'
           data to test the GAN implementation"""
        if False:
            #Single pixel, always in the same place:  SUCCESS
            self.data = np.zeros_like(self.data) - 1
            self.data[:,1] = 1

            #Vertical lines
            self.data = np.zeros_like(self.data) - 1
            for i in range(self.data.shape[0]):
                d = self.data[i, :].reshape((4,4))
                d[:,np.random.randint(0,4)] = 1
                self.data[i,] = d.flatten()


    def __new_epoch(self):
        """
        Reset the queue of shuffled indices that will be removed as we
        return examples.  Increment the epoch counter.
        """
        self.epoch += 1
        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)
        self.q = list(indices)

    def next_batch(self, batch_size):
        """Return batch_size examples in an array."""
        batch_data = np.zeros([batch_size,] + list(self.example_shape))
        for i in range(batch_size):
            index = self.q.pop()
            batch_data[i,...] = self.data[index]
            if len(self.q)==0:
                self.__new_epoch()

        return batch_data

