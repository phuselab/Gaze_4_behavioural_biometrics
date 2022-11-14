import numpy as np
import scipy.io as sio
from os.path import join
from os import listdir, remove
from zipfile import ZipFile
import pandas as pd
from my_utils.gaze import dva2pixels

def load_event_features(file):
    features = np.load(file, allow_pickle=True)
    n_ex = len(features)

    feat_fixs = []
    feat_sacs = []
    stim_fix = []
    stim_sac = []

    for e in range(n_ex):
        curr_data_dict = features[e]
        try:
            feat_fix = curr_data_dict['feat_fix']
            feat_sac = curr_data_dict['sacc_fix']

            feat_fixs.append(feat_fix)
            feat_sacs.append(feat_sac)
            stim_fix.append(np.repeat(curr_data_dict['stimulus'], len(feat_fix))[:,np.newaxis])
            stim_sac.append(np.repeat(curr_data_dict['stimulus'], len(feat_sac))[:,np.newaxis])
        except:
            continue
    feat_fixs = np.vstack(feat_fixs)
    feat_sacs = np.vstack(feat_sacs)
    stim_fix = np.vstack(stim_fix)
    stim_sac = np.vstack(stim_sac)

    return feat_fixs, feat_sacs, stim_fix, stim_sac

def load_george_features(file):
    features = np.load(file, allow_pickle=True)
    n_ex = len(features)

    feat_fixs = []
    feat_sacs = []
    stim_fix = []
    stim_sac = []

    for e in range(n_ex):
        curr_data_dict = features[e]
        feat_fix = curr_data_dict['feat_fix'].to_numpy()
        feat_sac = curr_data_dict['sacc_fix'].to_numpy()

        feat_fixs.append(feat_fix)
        feat_sacs.append(feat_sac)
        stim_fix.append(np.repeat(curr_data_dict['stimulus'], len(feat_fix))[:,np.newaxis])
        stim_sac.append(np.repeat(curr_data_dict['stimulus'], len(feat_sac))[:,np.newaxis])

    feat_fixs = np.vstack(feat_fixs)
    feat_sacs = np.vstack(feat_sacs)
    stim_fix = np.vstack(stim_fix)
    stim_sac = np.vstack(stim_sac)

    return feat_fixs, feat_sacs, stim_fix, stim_sac


def load_features_file(file):
    features_dict = sio.loadmat(file)

    data = np.transpose([features_dict['id'], features_dict['B00'], features_dict['B01'], features_dict['B11'],
                         features_dict['S00'], features_dict['S01'], features_dict['S11']])[0]


    return data

def load_features_file_new(file):
    features_dict = sio.loadmat(file)

    data = np.transpose([features_dict['id'], features_dict['B00'], features_dict['B01'], features_dict['B11'],
                         features_dict['S00'], features_dict['S01'], features_dict['S11'], 
                         features_dict['B00_s'], features_dict['B01_s'], features_dict['B11_s'],
                         features_dict['S00_s'], features_dict['S01_s'], features_dict['S11_s'], 
                         features_dict['mu_ig'], features_dict['loc_ig'], features_dict['scale_ig'],
                         features_dict['alpha'], features_dict['beta'],
                         features_dict['kappa'], features_dict['loc_vm'], features_dict['scale_vm']])[0]

    return data

def load_features_file_new_fixOnly(file):
    features_dict = sio.loadmat(file)

    data = np.transpose([features_dict['id'], features_dict['B00'], features_dict['B01'], features_dict['B11'],
                         features_dict['S00'], features_dict['S01'], features_dict['S11'],  
                         features_dict['mu_ig'], features_dict['loc_ig'], features_dict['scale_ig']])[0]

    return data

def load_features(path):
    global_data = []
    for file in listdir(path):
        features_dict = sio.loadmat(join(path, file))

        data = np.transpose([features_dict['id'], features_dict['B00'], features_dict['B01'], features_dict['B11'],
                             features_dict['S00'], features_dict['S01'], features_dict['S11']])[0]
        if len(data)<200: # if the dataset is coutrot
            data = data[:35]  # evens out data length
        global_data.append(data)
    return np.asarray(global_data)

def load_gazebase(path='datasets/GazeBase_v2_0', round='Round_9', session='S1', task='Video_1'):
    print("Unzipping data...")
    base_path = join(path, round)
    sub_zip_list = listdir(base_path)
    try:
        for sub_zip_file in sub_zip_list:
            if sub_zip_file[0] == '.':
                continue
            curr_zip_file = join(base_path, sub_zip_file)
            with ZipFile(curr_zip_file, 'r') as zipObj:
                # Extract all the contents
                new_dir = join(base_path, sub_zip_file[:-4])
                zipObj.extractall(new_dir)
                remove(curr_zip_file)
        print('Files extracted')
    except IsADirectoryError:
        print('Files already extracted')

    scanpath = []
    for sub_file in listdir(base_path):
        sub_scanpath = []
        if sub_file[0] == '.':
            continue
        sub_csv = listdir(join(base_path, sub_file, session, session+'_'+ task))[0]
        eye_dataframe = pd.read_csv(join(base_path, sub_file, session, session+'_'+task, sub_csv))
        eye_dataframe.dropna(subset=['x','y'], inplace=True)
        raw_eye_data = eye_dataframe[['x', 'y']].to_numpy()
        # Dividing data in sessions
        sec = 1
        fs = 1000.
        n_samples = int(sec * fs)
        for trial in range(raw_eye_data.shape[0] // (n_samples)):
            curr_gaze_ang = raw_eye_data[int(trial * n_samples):int((trial + 1) * n_samples)]
            # from angles to x,y coordinates
            distance = 0.55
            width = 0.474
            height = 0.297
            x_res = 1680
            y_res = 1050
            curr_gaze = dva2pixels(curr_gaze_ang, distance, width, height, x_res, y_res)
            sub_scanpath.append(curr_gaze)
        sub_scanpath = np.asarray(sub_scanpath)
        scanpath.append(sub_scanpath)
    scanpath = np.asarray(scanpath)
    parameters = {
        'distance':0.55,
        'width':0.474,
        'height':0.297,
        'x_res':1680,
        'y_res':1050,
        'fs':1000}
    return scanpath, parameters


def load_cerf(path='datasets/CerfDataset'):
    scanpath = []
    data = sio.loadmat(join(path,'fixations.mat'))['sbj'][0]
    for subject in range(8):
        sub_scanpath = []
        for session in range(200):
            recording_x = np.concatenate(data[subject]['scan'][0][0][:, session][0][0][0][3]).ravel()
            recording_y = np.concatenate(data[subject]['scan'][0][0][:, session][0][0][0][4]).ravel()
            recording_x = np.reshape(recording_x, (recording_x.shape[0], 1))
            recording_y = np.reshape(recording_y, (recording_y.shape[0], 1))
            sub_scanpath.append(np.concatenate((recording_x, recording_y), 1))
        sub_scanpath = np.asarray(sub_scanpath)
        scanpath.append(sub_scanpath)
    parameters = {
        'distance':0.8,
        'width':0.43,
        'height':0.31,
        'x_res': 1024,
        'y_res': 768,
        'fs': 1000.}
    scanpath = np.asarray(scanpath)
    return scanpath, parameters

def load_coutrot2(path='datasets/LondonMuseum_Ext'):
    scanpath = []
    for sub in listdir(path):
        sub_scanpath = []
        for trial in listdir(path+'/'+sub):            
            data = sio.loadmat(path+'/'+sub+'/'+trial)['xy']
            sub_scanpath.append(data)
        sub_scanpath = np.asarray(sub_scanpath)
        scanpath.append(sub_scanpath)
    parameters = {'distance':0.57,
                  'width':0.4,
                  'height':0.3,
                  'x_res':1280,
                  'y_res':1024,
                  'fs':250.}
    return np.asarray(scanpath), parameters

def load_coutrot(path='datasets/LondonMuseum_Ext'):
    data_dir_x = join(path, 'xcoord')
    data_dir_y = join(path, 'ycoord')
    scanpath = []
    for file in listdir(data_dir_x):
        sub_scanpath = []
        # just the first one for test reasons
        subject = int(file.split("b")[1].split("_")[0])
        
        # load the matlab file obtaining x and y coordinates eyes positions
        sub_data_x = sio.loadmat(join(data_dir_x, ("new_x_sub" + str(subject) + "_xcoord.mat")))['array'][0]
        sub_data_y = sio.loadmat(join(data_dir_y, ("new_y_sub" + str(subject) + "_ycoord.mat")))['array'][0]

        for i in range(len(sub_data_x)):
            if len(sub_data_x[i]) <= 0:
                continue
            recording_x = sub_data_x[i]
            recording_y = sub_data_y[i]
            sub_scanpath.append(np.concatenate((recording_x, recording_y), 1))

        sub_scanpath = np.asarray(sub_scanpath[:35])
        scanpath.append(sub_scanpath)
    parameters = {
        'distance':0.57,
        'width':0.4,
        'height':0.3,
        'x_res':1280,
        'y_res':1024,
        'fs':250.}
    return np.asarray(scanpath), parameters

def load_dataset(name, path, round='Round_9', session='S1', task='Video_1'):
    if name=='FIFA':
        return load_cerf(path)
    elif name=='gazebase':
        return load_gazebase(path, round, session, task)
    elif name=='coutrot':
        return load_coutrot2(path)

def load_cerf_trial(path, sub, trial):
    mat_file = path + '/fixations.mat'
    data = sio.loadmat(mat_file)['sbj'][0]
    recording_x = np.concatenate(data[sub]['scan'][0][0][:, trial][0][0][0][3]).ravel()
    recording_y = np.concatenate(data[sub]['scan'][0][0][:, trial][0][0][0][4]).ravel()
    recording_x = np.reshape(recording_x, (recording_x.shape[0], 1))
    recording_y = np.reshape(recording_y, (recording_y.shape[0], 1))
    curr_gaze = np.concatenate((recording_x, recording_y), 1)
    return curr_gaze

def load_coutrot_trial(path, sub, trial):
    mat_file = path + sub + '/' + trial
    curr_gaze = sio.loadmat(mat_file)['xy']
    return curr_gaze

def load_zuco_trial(path, sub, trial):
    mat_file = path + sub + '/' + trial
    data_temp = sio.loadmat(mat_file)['data']
    curr_gaze = np.transpose(np.vstack((data_temp[:, 1], data_temp[:, 2])))
    return curr_gaze


def load_trial(dataset_name, path, sub, trial):
    if dataset_name=="coutrot":
        return load_coutrot_trial(path, sub, trial)
    elif dataset_name=="cerf":
        return load_cerf_trial(path, sub, trial)
    elif dataset_name=="zuco":
        return load_zuco_trial(path, sub, trial)
    else:
        print("Dataset not recognized")

if __name__ == '__main__':
    data_cerf = load_cerf('../datasets/CerfDataset')
    data_coutrot = load_coutrot('../datasets/LondonMuseum_Ext')
    data_gazebase = load_gazebase('../datasets/GazeBase_v2_0')