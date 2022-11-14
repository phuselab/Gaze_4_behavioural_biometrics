clear
close all

load('fixations.mat')

males = 0;
females = 0;

for sub_n = 1:length(sbj)
    n_trials = size(sbj{1,sub_n}.scan,2);
    data = cell(1,n_trials);
    
    if strcmp(sbj{1,sub_n}.sex, 'm')
        males = males + 1;
    else
        females = females + 1;
    end
    
    for t=1:n_trials
        scanX = sbj{1,1}.scan{1,t}.scan_x;
        scanY = sbj{1,1}.scan{1,t}.scan_y;

        scan = [scanX, scanY];
                
        data{t} = scan;
        
    end
    
%     save(['sub_', num2str(sub_n), '.mat'], 'data')
end