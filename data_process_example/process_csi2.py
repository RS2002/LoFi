import numpy as np
import pickle


result=[]
pad=[-1000]*52
loacl_gap=10000

def process_time(timestamp):
    t = timestamp.split()
    t = t[-1].split(":")
    h = float(t[0])
    m = float(t[1])
    t = t[-1].split(".")
    s = float(t[0])
    ms = float(t[1])
    return h * 60 * 60 * 100 + m * 60 * 100 + s * 100 + ms


with open("./csi_data.pkl", 'rb') as f:
    csi = pickle.load(f)

for data in csi:
    csi_time=data['csi_time']
    local_time=data['csi_local_time']
    magnitude=data['magnitude']
    phase=data['phase']
    people=data['people']

    last_local=None
    last_glob=None
    current_magnitude=[]
    current_phase=[]
    current_timestamp=[]
    global_timestamp=[]
    for i in range(len(csi_time)):
        if last_local is None:
            last_local=local_time[i]
            last_glob=process_time(csi_time[i])
            current_magnitude.append(magnitude[i])
            current_phase.append(phase[i])
            current_timestamp.append(local_time[i])
        else:
            local = local_time[i]
            glob = process_time(csi_time[i])
            num=round((local-last_local-loacl_gap)/loacl_gap)
            if num>0:
                delta=(local-last_local)/(num+1)
                delta_glob=(glob-last_glob)/(num+1)
                for j in range(num):
                    current_magnitude.append(pad)
                    current_phase.append(pad)
                    current_timestamp.append(current_timestamp[-1] + delta)
                    global_timestamp.append(global_timestamp[-1]+delta+glob)
            current_magnitude.append(magnitude[i])
            current_phase.append(phase[i])
            current_timestamp.append(local)
            global_timestamp.append(glob)
            last_local = local
            last_glob = glob

    print(len(current_magnitude))
    result.append({
        'time': np.array(current_timestamp),
        'global_time': np.array(global_timestamp),
        'people': people,
        'magnitude': np.array(current_magnitude),
        'phase': np.array(current_phase)
    })

output_file = './data_sequence.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(result, f)