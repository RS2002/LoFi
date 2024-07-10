from torch.utils.data import Dataset
import numpy as np
import pickle
import torch

pad = -1000

class CSI_dataset(Dataset):
    def __init__(self, magnitudes, phases, timestamp, x, y, num=2000, length=100):
        super().__init__()
        self.magnitudes = magnitudes
        self.phases = phases
        self.timestamp=timestamp
        self.x = x
        self.y = y
        self.num=num
        self.length=length

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        i=np.random.randint(0,len(self.magnitudes))
        magnitude_full = self.magnitudes[i]
        phase_full = self.phases[i]
        timestamp_full = self.timestamp[i]
        x_full = self.x[i]
        y_full = self.y[i]

        while magnitude_full.shape[0]<=self.length:
            i = np.random.randint(0, len(self.magnitudes))
            magnitude_full = self.magnitudes[i]
            phase_full = self.phases[i]
            timestamp_full = self.timestamp[i]
            x_full = self.x[i]
            y_full = self.y[i]

        l=np.random.randint(0,phase_full.shape[0]-self.length)
        r=l+self.length
        magnitude = magnitude_full[l:r]
        phase = phase_full[l:r]
        timestamp = timestamp_full[l:r]
        x = x_full[l:r]
        y = y_full[l:r]

        while pad in x:
            l = np.random.randint(0, phase_full.shape[0] - self.length)
            r = l + self.length
            magnitude = magnitude_full[l:r]
            phase = phase_full[l:r]
            timestamp = timestamp_full[l:r]
            x = x_full[l:r]
            y = y_full[l:r]
        return magnitude,phase,torch.tensor(x),torch.tensor(y),timestamp

def load_data(data_path="./data/wiloc.pkl", train_prop=0.9, train_num=2000, test_num=200, length=100):
    with open(data_path, 'rb') as f:
        csi = pickle.load(f)

    people_list = []
    timestamp = []
    magnitudes = []
    phases = []
    x_list = []
    y_list = []

    for data in csi:
        local_time = data['time']
        magnitude = data['magnitude']
        phase = data['phase']
        people = data['people']
        x = data['x']
        y = data['y']

        people_list.append(people)
        magnitudes.append(magnitude)
        timestamp.append(local_time)
        phases.append(phase)
        x_list.append(x)
        y_list.append(y)

    if train_prop is None:
        return CSI_dataset(magnitudes, phases, timestamp, x_list, y_list, num=train_num, length=length)
    else:
        train_timestamp = []
        train_magnitudes = []
        train_phases = []
        train_x = []
        train_y = []
        test_timestamp = []
        test_magnitudes = []
        test_phases = []
        test_x = []
        test_y = []
        for i in range(len(people_list)):
            num=magnitudes[i].shape[0]
            num=int(num*train_prop)

            train_timestamp.append(timestamp[i][:num])
            train_magnitudes.append(magnitudes[i][:num])
            train_phases.append(phases[i][:num])
            train_x.append(x_list[i][:num])
            train_y.append(y_list[i][:num])

            test_timestamp.append(timestamp[i][num:])
            test_magnitudes.append(magnitudes[i][num:])
            test_phases.append(phases[i][num:])
            test_x.append(x_list[i][num:])
            test_y.append(y_list[i][num:])
        return CSI_dataset(train_magnitudes, train_phases, train_timestamp, train_x, train_y, num=train_num, length=length), CSI_dataset(test_magnitudes, test_phases, test_timestamp, test_x, test_y, num=test_num, length=length)

