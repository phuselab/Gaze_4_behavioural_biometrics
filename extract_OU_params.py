import numpy as np
from multiprocessing import Pool, cpu_count
from my_utils.saver import save_event_features
from my_utils.gaze import split_events, pixels2angles, get_fixndur, angle_between_first_and_last_points
from my_utils.loader import load_dataset
import matplotlib.pyplot as plt
import pymc3 as pm
from OrnsteinUhlenbeckPyMC.EU import Mv_EulerMaruyama
import theano
import theano.tensor as tt
from scipy.stats import iqr
import nslr_hmm

def get_xy_features(xy, sampleRate, type_event):
    duration = xy.shape[0] / sampleRate  # calculate each event duration
    if type_event == 'sac':
        angle = angle_between_first_and_last_points(xy)
        ampl = np.linalg.norm(xy[0,:] - xy[-1,:])
        return angle, ampl, duration
    else:
        return duration


dataset_name = 'FIFA'
lib = 'pymc'
method = 'SVI'
dataset_path = 'datasets/FIFA'

save_trace = False

#OrnsteinUhlenbeckPyMC SDE
def sde(xt, B, U, SIGMA):
    dif = U-xt
    res = tt.dot(B, dif.T)
    return res.T, SIGMA

fs = 1000.

data_np = np.random.randn(10,2)
data_th = theano.shared(data_np)
with pm.Model() as model:
    print('\n\tBuilding Model...')
    # LKJ Prior over the "covariance matrix" Beta
    packed_LB = pm.LKJCholeskyCov('packed_LB', n=2, eta=2, sd_dist=pm.HalfCauchy.dist(2.5))
    LB = pm.expand_packed_triangular(2, packed_LB)
    B = pm.Deterministic('B', LB.dot(LB.T))

    U = np.zeros(2)

    # LKJ Prior over the "covariance matrix" Gamma
    packed_LS = pm.LKJCholeskyCov('packed_LS', n=2, eta=2, sd_dist=pm.HalfCauchy.dist(2.5))
    LS = pm.expand_packed_triangular(2, packed_LS)
    SIGMA = pm.Deterministic('SIGMA', LS.dot(LS.T))

    # Multi-variate Euler Maruyama
    X = Mv_EulerMaruyama('X', 1/fs, sde,
                          (B, U, SIGMA,), shape=(data_th.shape.eval()),
                          testval=data_th, observed=data_th)

def extract_features_sub(sub_data, sub, parameters, lib, method, dset):
    '''
    Extract and save the features of sub-th subject
    :param sub_data: data of the sub-th subject
    :param sub: subject index
    :param parameters: screen parameters
    :param lib: library used for the inference
    :param method: maximum a posteriori estimation or stochastic variational inference
    :return: None
    '''
    print('\nSubject number', sub+1)
    all_features = []

    # Dividing data in sessions
    for session, gaze_data in enumerate(sub_data):
        print('\n\tSession number', session+1, '/', len(sub_data))

        n_samples = gaze_data.shape[0]
        dur = n_samples / fs
        t = np.linspace(0., dur, n_samples)
        gaze_data_ang = pixels2angles(gaze_data, parameters['distance'], parameters['width'],
                        parameters['height'], parameters['x_res'], parameters['y_res'])
        print('\nStarting NSLR Classification...')
        sample_class, segmentation, seg_class = nslr_hmm.classify_gaze(t, gaze_data_ang)
        print('...done. Starting CBW Estimation!')
        fixations = sample_class == nslr_hmm.FIXATION
        sp = sample_class == nslr_hmm.SMOOTH_PURSUIT
        saccades = sample_class == nslr_hmm.SACCADE
        pso = sample_class == nslr_hmm.PSO
        fix = np.logical_or(fixations, sp).astype(bool)  # merge fixations and smooth pursuits
        sac = np.logical_or(saccades, pso).astype(bool) #merge saccades and pso
        all_fix = split_events(gaze_data, fix)
        all_sac = split_events(gaze_data, sac)

        print('\tStarting CBW Estimation!')

        features = {}
        traces_fix = []
        traces_sac = []

        feature_fix = []
        for fi, curr_fix in enumerate(all_fix):
            print('\tProcessing Fixation ' + str(fi+1) + ' of ' + str(len(all_fix)))
            try:
                fdur = get_xy_features(curr_fix, fs, 'fix')

                with model:
                    # Switch out the observed dataset
                    data_th.set_value(curr_fix)
                    approx = pm.fit(n=20000, method=pm.ADVI())
                    trace_fix = approx.sample(draws=10000)
                    B_fix = trace_fix['B'].mean(axis=0)
                    Sigma_fix = trace_fix['SIGMA'].mean(axis=0)
                    B_fix_sd = iqr(trace_fix['B'], axis=0)
                    Sigma_fix_sd = iqr(trace_fix['SIGMA'], axis=0)
            
            except:
                print('\tSomething went wrong with feature extraction... Skipping fixation')

            curr_f_fix = np.array([B_fix[0, 0], B_fix[0, 1], B_fix[1, 1],
                                    B_fix_sd[0, 0], B_fix_sd[0, 1], B_fix_sd[1, 1],
                                    Sigma_fix[0, 0], Sigma_fix[0, 1], Sigma_fix[1, 1],
                                    Sigma_fix_sd[0, 0], Sigma_fix_sd[0, 1], Sigma_fix_sd[1, 1],
                                    fdur])
            feature_fix.append(curr_f_fix)
            tf = {}
            tf['B'] = trace_fix['B']
            tf['S'] = trace_fix['SIGMA']
            traces_fix.append(tf)

        features_fix = np.vstack(feature_fix)
        

        feature_sac = []   
        for si,curr_sac in enumerate(all_sac):
            if len(curr_sac) < 4:
                continue
            print('\tProcessing Saccade ' + str(si+1) + ' of ' + str(len(all_sac)))
            try:
                angle, ampl, sdur = get_xy_features(curr_sac, fs, 'sac')
                with model:
                    # Switch out the observed dataset
                    data_th.set_value(curr_sac)
                    approx = pm.fit(n=20000, method=pm.ADVI())
                    trace_sac = approx.sample(draws=10000)
                    B_sac = trace_sac['B'].mean(axis=0)
                    Sigma_sac = trace_sac['SIGMA'].mean(axis=0)
                    B_sac_sd = iqr(trace_sac['B'], axis=0)
                    Sigma_sac_sd = iqr(trace_sac['SIGMA'], axis=0)

            except:
                print('\tSomething went wrong with feature extraction... Skipping saccade')
            
            curr_f_sac = np.array([B_sac[0, 0], B_sac[0, 1], B_sac[1, 1],
                                    B_sac_sd[0, 0], B_sac_sd[0, 1], B_sac_sd[1, 1],
                                    Sigma_sac[0, 0], Sigma_sac[0, 1], Sigma_sac[1, 1],
                                    Sigma_sac_sd[0, 0], Sigma_sac_sd[0, 1], Sigma_sac_sd[1, 1],
                                    angle, ampl, sdur])
            feature_sac.append(curr_f_sac)
            tf = {}
            tf['B'] = trace_sac['B']
            tf['S'] = trace_sac['SIGMA']
            traces_sac.append(tf)

        features_sac = np.vstack(feature_sac)

        features['label'] = float(sub)
        features['stimulus'] = session
        features['feat_fix'] = features_fix
        features['sacc_fix'] = features_sac
        features['traces_fix'] = traces_fix
        features['traces_sac'] = traces_fix

        all_features.append(features)

    save_event_features(all_features, dataset_name, "event_features_" + str(sub), type='OU_posterior', method='VI', dset=dset)


    return 'Features saved for subject number ' + str(sub+1)

def get_all_features(data, parallel=False):
    '''
    Parallelize features extraction
    :param data: dataset
    :return: None
    '''

    n_ex_per_sub = data[0].shape[0]
    n_train = int(n_ex_per_sub * 0.75)

    if parallel:
        n_processes = min(cpu_count(), len(data))
        print('The computation will be parallelized in ', n_processes, ' processes')
        # sub_file in listdir(base_path)
        with Pool(n_processes) as p:
            multiple_results = [p.apply_async(extract_features_sub, args=(sub_data[:n_train], sub, parameters, lib, method, 'train')) for sub, sub_data in enumerate(data)]
            _ = [res.get() for res in multiple_results]

        print('\n\nTest data!!\n\n')

        with Pool(n_processes) as p:
            multiple_results = [p.apply_async(extract_features_sub, args=(sub_data[n_train:], sub, parameters, lib, method, 'test')) for sub, sub_data in enumerate(data)]
            _ = [res.get() for res in multiple_results]
    else:
        
        for sub, sub_data in enumerate(data):
            extract_features_sub(sub_data[:n_train], sub, parameters, lib, method, dset='train')
            extract_features_sub(sub_data[n_train:], sub, parameters, lib, method, dset='test')

if __name__ == "__main__":
    print('Dataset name:', dataset_name, '\nDataset path: ', dataset_path)
    data, parameters = load_dataset(dataset_name, dataset_path)

    get_all_features(data)