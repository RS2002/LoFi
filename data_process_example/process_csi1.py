import numpy as np
import pickle
import os
import pandas as pd

root="./csi"
data=[]
csi_vaid_subcarrier_index = range(0, 52)

def handle_complex_data(x, valid_indices):
    real_parts = []
    imag_parts = []
    for i in valid_indices:
        real_parts.append(x[i * 2])
        imag_parts.append(x[i * 2 - 1])
    return np.array(real_parts) + 1j * np.array(imag_parts)

people_id=0
for people in os.listdir(root):
    print(people)
    path=os.path.join(root,people)

    for file in os.listdir(path):
        if file[-3:] != "csv":
            continue
        print(file)
        df = pd.read_csv(os.path.join(path,file))
        df.dropna(inplace=True)
        df['data'] = df['data'].apply(lambda x: eval(x))
        complex_data = df['data'].apply(lambda x: handle_complex_data(x, csi_vaid_subcarrier_index))
        magnitude = complex_data.apply(lambda x: np.abs(x))
        phase = complex_data.apply(lambda x: np.angle(x, deg=True))
        time = np.array(df['timestamp'])
        local_time = np.array(df['local_timestamp'])

        data.append({
            'csi_time':time,
            'csi_local_time':local_time,
            'people_name': people,
            'people': people_id,
            'magnitude': np.array([np.array(a) for a in magnitude]),
            'phase': np.array([np.array(a) for a in phase]),
            'CSI': np.array([np.array(a) for a in complex_data])
        })
    people_id+=1


output_file = './csi_data.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(data, f)