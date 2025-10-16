import numpy as np


def parse_data(filename, random=False):
    f = open(filename)
    data = f.readlines()
    tgts = []
    datapoints = []
    features = []
    newdata = False
    for line in data:
        line = line.strip()
        vals = line.split(",")
        if newdata and len(vals) >= 3 and len(features) == 10:
            #print(features)
            datapoints.append(features)
            newdata = False
            features = []

        if len(vals) == 5 or len(vals) == 6:
            #print(vals)
            tgts.append(vals)
        if len(vals) == 3:
            features.append(vals)
        else:
            newdata = True


    datapoints.append(features)


    datapoints = np.array(datapoints).astype(float)
    flattened_data = datapoints.reshape(datapoints.shape[0], 30)
    tgts = np.array(tgts).astype(float)

    if tgts.shape[1] == 5:
        angles = tgts[:,1]
        sin = np.sin(angles)
        cos = np.cos(angles)

        new_tgts = np.hstack((tgts[:, :1], sin.reshape(-1, 1), cos.reshape(-1, 1), tgts[:, 2:]))
        #new_tgts = np.hstack((new_tgts, angles.reshape(-1, 1)))
    else:
        new_tgts = tgts


    flat = datapoints.flatten()

    combined_data = list(zip(flattened_data, new_tgts))
    if random:
        random.shuffle(combined_data)

    # Unzip the shuffled pairs back into separate arrays
    shuffled_flattened_data, shuffled_new_tgts = zip(*combined_data)
    shuffled_flattened_data = np.array(shuffled_flattened_data)
    shuffled_new_tgts = np.array(shuffled_new_tgts)

    #print(shuffled_flattened_data.shape)
    #print(shuffled_new_tgts.shape)
    return shuffled_flattened_data, shuffled_new_tgts  

def parse_data_simple(filename, invert_pt=False):
    f = open(filename)
    data = f.readlines()
    tgts = []
    datapoints = []
    features = []
    newdata = False
    for line in data:
        line = line.strip()
        vals = line.split(",")
        if newdata and len(vals) >= 3 and len(features) == 10:
            #print(features)
            datapoints.append(features)
            newdata = False
            features = []

        if len(vals) == 5 or len(vals) == 6:
            #print(vals)
            tgts.append(vals)
        if len(vals) == 3:
            features.append(vals)
        else:
            newdata = True


    datapoints.append(features)


    datapoints = np.array(datapoints).astype(float)
    flattened_data = datapoints.reshape(datapoints.shape[0], 30)
    tgts = np.array(tgts).astype(float)

    if invert_pt == True:
        tgts[:,2] = 1.0 / tgts[:,2]

    return flattened_data, tgts


def reduce_dim(input_arr):

    sin_preds = input_arr[:,1]
    cos_preds = input_arr[:,2]

    pred_angle = np.arctan2(sin_preds, cos_preds)
    pred_angle = np.mod(pred_angle, 2 * np.pi)

    output = np.delete(input_arr, [1, 2], axis=1)
    output = np.insert(output, 1, pred_angle, axis=1)
    return output

def invert_idx(helix_params, idx):

    inverted = helix_params.copy()

    inverted[:, idx] = 1 / inverted[:, idx]

    return inverted

def parse_data_ls_realistic(filename):
    f = open(filename)
    data = f.readlines()
    tgts = []
    chisq_vals = []
    newdata = False
    for line in data:
        line = line.strip()
        vals = line.split(",")      
        if len(vals) == 6:
            #print(vals)
            tgts.append(vals[1:])
            chisq_vals.append(vals[0])


    tgts = np.array(tgts).astype(float)
    tgts[:,2] = 1.0 / tgts[:,2]
    chisq_vals = np.array(chisq_vals).astype(float)
    return chisq_vals, tgts

def parse_data_ls_idealized(filename):
    f = open(filename)
    data = f.readlines()
    tgts = []
    chisq_vals = []
    newdata = False
    for line in data:
        line = line.strip()
        vals = line.split(",")      
        if len(vals) == 9:
            #print(vals)
            tgts.append(vals[4:])
            chisq_vals.append(vals[:4])


    tgts = np.array(tgts).astype(float)
    tgts[:,2] = 1.0 / tgts[:,2]
    chisq_vals = np.array(chisq_vals).astype(float)
    return chisq_vals, tgts