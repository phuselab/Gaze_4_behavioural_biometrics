from math import atan2, degrees
import numpy as np
import nslr_hmm
import math

def angle_between_first_and_last_points(xy):
    """ Angle between two points in degrees
    Input:
    xy: ndarray
        2D array of gaze points (x,y)
    Return: float
        angle between the first and last point
    """
    assert len(xy.shape) == 2 and xy.shape[1] == 2
    assert xy.shape[0] >= 2  # check if the input has at least 2 rows

    point_a = xy[0]
    point_b = xy[-1]
    diff_x = point_b[0] - point_a[0]
    diff_y = point_b[1] - point_a[1]

    return math.degrees(math.atan2(diff_x, diff_y))

def dva2pixels(angles, d, w, h, x_res, y_res):
    cm_size = angles * (d*100) * 0.017455
    # traslation
    cm_size += [w*100/2, h*100/2]
    pixels = cm_size * x_res / (w*100)
    return pixels

def pixels2angles(xs, dist, w, h, resx, resy):
	#xs: Nx2 data
	#dist: distance from screen in meters
	#w, h: width and height of the sceen (m)
	#resx, resy: resoluion of the screen

	screenmeters = xs / np.array([resx, resy])  #procedura standard
	screenmeters -= 0.5 #per avere 0,0 al centro dello schermo
	screenmeters *= np.array([w, h])
	angles = np.degrees(np.arctan(screenmeters / dist))
	return angles

def split_events(gaze, bool_mask, check_event_quality=True):
	bool_mask = bool_mask.astype(int)
	if bool_mask[0] == 1:
		bool_mask = np.insert(bool_mask, 0, 0)
	if bool_mask[-1] == 1:
		bool_mask = np.append(bool_mask, 0)

	starts_event = np.where(np.diff(bool_mask) == 1)[0]
	ends_event = np.where(np.diff(bool_mask) == -1)[0]
	n_events = starts_event.shape[0]
	all_events = []

	for f in range(n_events):
		curr_event_idx = np.zeros(gaze.shape[0])
		curr_event_idx[starts_event[f]:ends_event[f]] = 1
		curr_event_idx = curr_event_idx.astype(bool)
		if np.sum(curr_event_idx) < 4:
			continue
		all_events.append(gaze[curr_event_idx,:])

	return all_events

def prepare_events(events, type_ev):
	result = []
	if type_ev == 'fix':
		for e in events:
			e_new = e - np.mean(e, axis=0)
			result.append(e_new)
		return result
	elif type_ev == 'sac':
		for e in events:
			e_new = e - e[-1]
			result.append(e_new)
		return result
	else:
		raise ValueError('Unrecognized type of event! Should be "fix" or "sac"')

def prepare_events_shared(events, type_ev):
	result = []
	lenghts = []

	if type_ev == 'fix':
		for e in events:
			e_new = e - np.mean(e, axis=0)
			result.append(e_new)
			lenghts.append(len(e_new))
		max_len = max(lenghts)
		n_ev = len(lenghts)
		enp = np.empty((n_ev, max_len, 2))
		enp[:] = np.nan
		for i,e in enumerate(result):
			enp[i,:len(e),:] = e
		return enp
	
	elif type_ev == 'sac':
		for e in events:
			e_new = e - e[-1]
			result.append(e_new)
			lenghts.append(len(e_new))
		max_len = max(lenghts)
		n_ev = len(lenghts)
		enp = np.empty((n_ev, max_len, 2))
		enp[:] = np.nan
		for i,e in enumerate(result):
			enp[i,:len(e),:] = e
		return enp
	
	else:
		raise ValueError('Unrecognized type of event! Should be "fix" or "sac"')


def get_fixndur(gaze_data, sample_class, fs):
	fixations = []
	fix_bool = sample_class == nslr_hmm.FIXATION
	sp = sample_class == nslr_hmm.SMOOTH_PURSUIT
	fix = np.logical_or(fix_bool, sp).astype(int)	#merge fixations and smooth pursuits
	if fix[0] == 1:
		fix = np.insert(fix, 0, 0)
	if fix[-1] == 1:
		fix = np.append(fix, 0)
	starts_fix = np.where(np.diff(fix) == 1)[0]
	ends_fix = np.where(np.diff(fix) == -1)[0]
	fix_durations = ends_fix - starts_fix
	fix_durations_sec = (fix_durations/fs)*1000		#milliseconds
	for i in range(starts_fix.shape[0]):
		fixations.append(np.mean(gaze_data[starts_fix[i]:ends_fix[i], :], axis=0))
	fixations = np.array(fixations)
	fix_plus_dur = np.hstack((fixations, np.expand_dims(fix_durations_sec, axis=1))).astype(int)

	return fix_plus_dur

def get_decision_stats(class_scan):
	dirs = []
	amps = []
	durs = []
	#directions, durations and amplitudes

	curr_scan = class_scan[:,0:2]
	nfix = curr_scan.shape[0]

	durs = class_scan[:,2]

	for f in range(nfix-1):
		curr_fix2 = curr_scan[f+1,:]
		curr_fix1 = curr_scan[f,:]
		direction = np.arctan2(curr_fix2[1]-curr_fix1[1], curr_fix2[0]-curr_fix1[0]) + np.pi
		amplitude = np.linalg.norm(curr_fix2-curr_fix1)
		dirs.append(direction)
		amps.append(amplitude)	

	return np.array(amps), np.array(dirs), durs