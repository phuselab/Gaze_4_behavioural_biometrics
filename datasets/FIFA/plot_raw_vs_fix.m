close all
clf
load fixations.mat;
load imgList.mat;
load annotations.mat;

%imnum=68;
subnum=6;
IMG_DIR = './faces/';

% imgname='0113.tif'
imgname='0028.tif'

imnum=1;

while ~strcmp(imgname,imgList{imnum})
    
    fprintf('\n Name %s : index= %d', imgList{imnum}, imnum)
    imnum=imnum+1;
end    
    

imagein=[IMG_DIR imgList{imnum}]

x = sbj{subnum}.scan{imnum}.scan_x + 10;
y = sbj{subnum}.scan{imnum}.scan_y + 10;

I = imread(imagein);

figure
imshow(I)

figure
% plot the raw data
% plot(x,y,'b-','LineWidth',3);

xds = x(1:50:end);
yds = y(1:50:end);
plot(xds,yds,'-x');

figure
% plot the raw data
% plot(x,y,'b-','LineWidth',3);

xds = x(1:50:end);
yds = y(1:50:end);
plot(xds,yds,'-x');
% hold on
s=subnum;


% x=sbj{s}.scan{imnum}.fix_x;
% y=sbj{s}.scan{imnum}.fix_y;
% dur=sbj{s}.scan{imnum}.fix_duration;
% for i=1:length(x)
%     plot(x(i),y(i),'r.', 'MarkerSize',dur(i)/10);
%     hold on;
% end
% hold on;
% plot(sbj{subnum}.scan{imnum}.fix_x,sbj{subnum}.scan{imnum}.fix_y,'r-','LineWidth',2);
