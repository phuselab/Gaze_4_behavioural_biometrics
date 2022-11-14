# Files description

## fixations.mat

Collects the scanpath of 8 subjects during the free-viewing of 200 images for 2 seconds each.

For each subjects there are:
- age
- sex
- scan: 200 scanpath recordings, one for each stimulus (image)
- response: 200 task-related labels
- order: 200 ordered stimuli id

Each scanpath recording contains:
- scan_x: 2021 x-coordinates of gaze 
- scan_y: 2021 y-coordinates of gaze
- fix_x: x-coordinates of fixations (variable lenght)
- fix_y: y-coordinates of fixations (variable lenght)
- fix_duration: durations of fixations in ms (variable lenght)

## subjects/sbj_*.mat

Collects data of 4 different task-related experiments.