import pickle

with open("./data_sequence.pkl", 'rb') as f:
    data_csi = pickle.load(f)
with open("./cv_data.pkl", 'rb') as f:
    data_cv = pickle.load(f)

data = []
pad = -1000
for i in range(len(data_csi)):
    csi = data_csi[i]
    cv = data_cv[i]

    x = cv['x']
    y = cv['y']
    img_path = cv['img_path']
    time_cv = cv['timestamp']

    csi_time = csi['global_time']
    local_time = csi['time']
    magnitude = csi['magnitude']
    phase = csi['phase']
    people = csi['people']

    x_list = []
    y_list = []
    path_list = []

    i = 0
    j = 0
    while csi_time[i] < time_cv[j]:
        i += 1
        x_list.append(pad)
        y_list.append(pad)
        path_list.append(pad)

    while i < len(csi_time):
        while csi_time[i] > time_cv[j]:
            j += 1
            if j > len(time_cv):
                break
        if j > len(time_cv):
            break
        x_list.append(x[j])
        y_list.append(y[j])
        path_list.append(img_path[j])
        i += 1

    if len(x_list) < len(csi_time):
        num = len(csi_time) - len(x_list)
        x_list = x_list + [pad] * num
        y_list = y_list + [pad] * num
        path_list = path_list + [pad] * num

    data.append({
        'magnitude': magnitude,
        'phase': phase,
        'x': x_list,
        'y': y_list,
        'img_path': path_list,
        'time': local_time,
        'people': people
    })
    print(people)
    print(len(magnitude),len(phase),len(x_list),len(y_list),len(path_list),len(local_time))

output_file = './wiloc.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(data, f)
