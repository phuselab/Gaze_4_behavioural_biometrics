
close all
% load the data 
clf
load fixations.mat;
load imgList.mat;
load annotations.mat;

%imnum=68;
subnum=2;

%IMG_DIR='/Volumes/VERBATIM_HD/Datasets/EyeTCerf/SMVJ/faces/';
IMG_DIR = './faces/';
%Fino a 247
%imgname='0106.tif'
imgname='0113.tif'

% imglist = dir([IMG_DIR '*.tif']);	% edit path/filename to match your images' location
% fnum    = length(imglist)

% for imnum=1:length(imgList)
%  fprintf('\n TrueName %s : index= %d', imgList{imnum}, imnum)
%  fprintf('\n Name %s : index= %d', imglist(imnum).name, imnum)
% end
% pause

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

% plot the fixations (subject 2 - chosen arbitrarily)
s=subnum;
%numallsubj=length(sbj)
%for s=1:numallsubj
    x=sbj{s}.scan{imnum}.fix_x;
    y=sbj{s}.scan{imnum}.fix_y;
    dur=sbj{s}.scan{imnum}.fix_duration;
    for i=1:length(x)
    plot(x(i),y(i),'y.', 'MarkerSize',dur(i)/10);
    hold on;
    %plot(px(k),py(k),'.','Color',col,'MarkerSize',fix_dur(k)*50);
    end
    hold on;
    plot(sbj{subnum}.scan{imnum}.fix_x,sbj{subnum}.scan{imnum}.fix_y,'y-','LineWidth',2);
%end
% plot the raw data 
 plot(sbj{subnum}.scan{imnum}.scan_x,sbj{subnum}.scan{imnum}.scan_y,'r-','LineWidth',3);