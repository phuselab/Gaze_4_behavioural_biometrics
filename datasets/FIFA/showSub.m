close all
% load the data 
clf
load fixations.mat;
load imgList.mat;
load annotations.mat;


IMG_DIR = './faces/';
SUB_DIR = './subjects/';



load('sbj_1.mat')

order=experiments{1,1}.order

%seleziona prima immagine osservata dal soggetto
im_num=order(1)

imgname = ['00' int2str(im_num) '.tif']

imnum=1;

while ~strcmp(imgname,imgList{imnum})
    
    fprintf('\n Name %s : index= %d', imgList{imnum}, imnum)
    imnum=imnum+1;
end    
    
%imnum=imnum-1;

%pause

imagein=[IMG_DIR imgList{imnum}]

% plot the image (image 16 - chosen arbitrarily) 
%img = imread(imgList{imnum});
img = imread(imagein);
imagesc(img);
hold on;

fix_x = experiments{1,1}.scan{1,1}.fix_x;
fix_y = experiments{1,1}.scan{1,1}.fix_y;
fix_starts = experiments{1,1}.scan{1,1}.fix_starts;
fix_stops = experiments{1,1}.scan{1,1}.fix_stops;
fix_duration = experiments{1,1}.scan{1,1}.fix_duration;
scan_x = experiments{1,1}.scan{1,1}.scan_x;
scan_y = experiments{1,1}.scan{1,1}.scan_y;
scan_t = experiments{1,1}.scan{1,1}.scan_t;



for i=1:length(fix_x)
    plot(fix_x(i),fix_y(i),'y.', 'MarkerSize',fix_duration(i)/10);
    hold on;
    %plot(px(k),py(k),'.','Color',col,'MarkerSize',fix_dur(k)*50);
end
hold on;
plot(fix_x,fix_y,'y-','LineWidth',2);
hold on;
% plot the raw data %end
plot(scan_x,scan_y,'r-','LineWidth',3);


% coloro nei raw data i punti che sono stati classificati come appartenenti
% a una fissazione
hold on;
index_starts = [];
index_stops  = [];

for j=1:length(fix_starts)
    
    ind_starts=find(scan_t == fix_starts(j))
    if(ind_starts  ~= 0)
        index_starts = [index_starts ind_starts];
        ind_stops=find(scan_t == fix_stops(j));
        if(ind_stops  ~= 0)
         index_stops = [index_stops ind_stops];
        else
        index_starts=index_starts(1:end-1);    
        end
%     ind_stops=find(scan_t == fix_stops(j));
%     if(ind_stops)
%         index_stops = [index_stops ind_stops];
%     end
        
    end
end
% mette a verde i punti raw inclusi nella fissazione
for j=1:length(index_starts)
    
    plot(scan_x(index_starts(j):index_stops(j)),scan_y(index_starts(j):index_stops(j)),'g-','LineWidth',4);
    hold on;
end
    

% plot(sbj{subnum}.scan{imnum}.scan_x,sbj{subnum}.scan{imnum}.scan_y,'r-','LineWidth',3);