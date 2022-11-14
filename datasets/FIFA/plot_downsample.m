close all
clf
load fixations.mat;
load imgList.mat;
load annotations.mat;

subnum=6;
IMG_DIR = './faces/';

imgname='0028.tif';

imnum=1;
while ~strcmp(imgname,imgList{imnum})
    
    fprintf('\n Name %s : index= %d', imgList{imnum}, imnum)
    imnum=imnum+1;
end    
    

imagein=[IMG_DIR imgList{imnum}]

x = sbj{subnum}.scan{imnum}.scan_x + 10;
y = sbj{subnum}.scan{imnum}.scan_y + 10;

rng(1);  % For reproducibility
n_samples = 50;
r = trnd(0.5,n_samples,2);
x_levy(1) = 512;
y_levy(1) = 384;
for i=2:size(r,1)
    x_levy(i) = x_levy(i-1) + r(i,1);
    y_levy(i) = y_levy(i-1) + r(i,2);
end


I = imread(imagein);


% imshow(I)

% imshow(rgb2gray(I))
% hold on
xds50 = x(1:50:end);
yds50 = y(1:50:end);
% plot(xds50,yds50,'-o','LineWidth',2, 'MarkerSize', 6, 'MarkerFaceColor','b');
% set(gca,'visible','off')
% 
% figure
% subplot(1,2,1)
% plot(xds50,yds50,'-o','LineWidth',2, 'MarkerSize', 6, 'MarkerFaceColor','b');
% grid on
% set(gca,'visible','off')
% set(gcf,'color','w');
% subplot(1,2,2)
% plot(x_levy,y_levy,'-o','LineWidth',2, 'MarkerSize', 6, 'MarkerFaceColor','b')
% grid on
% set(gca,'visible','off')
% set(gcf,'color','w');
% 
% figure
% subplot(1,2,1)
% xds = x(1:10:end);
% yds = y(1:10:end);
% plot(xds,yds,'-o','LineWidth',2, 'MarkerSize', 6, 'MarkerFaceColor','b');
% grid on
% set(gca,'visible','off')
% set(gcf,'color','w');
% subplot(1,2,2)
% plot(x_levy,y_levy,'-o','LineWidth',2, 'MarkerSize', 6, 'MarkerFaceColor','b')
% grid on
% set(gca,'visible','off')
% set(gcf,'color','w');
% 
% figure
% subplot(1,2,1)
% plot(x,y,'-o','LineWidth',2, 'MarkerSize', 2, 'MarkerFaceColor','b');
% hold on
% plot(xds50,yds50,'bo','LineWidth',2, 'MarkerSize', 6, 'MarkerFaceColor','b');
% grid on
% set(gca,'visible','off')
% set(gcf,'color','w');
% subplot(1,2,2)
% plot(x_levy,y_levy,'-o','LineWidth',2, 'MarkerSize', 6, 'MarkerFaceColor','b')
% grid on
% set(gca,'visible','off')
% set(gcf,'color','w');

figure
plot(x,y,'-o','LineWidth',4, 'MarkerSize', 4, 'MarkerFaceColor','b');
hold on
plot(xds50,yds50,'bo','LineWidth',4, 'MarkerSize', 8, 'MarkerFaceColor','b');
set(gca,'visible','off')
set(gcf,'color','w');

figure
plot(x,y,'-o','LineWidth',4, 'MarkerSize', 8, 'MarkerFaceColor','b');
set(gca,'visible','off')
set(gcf,'color','w');

figure
plot(x,y,'-o','LineWidth',4, 'MarkerSize', 8, 'MarkerFaceColor','b');
set(gca,'visible','off')
set(gcf,'color','w');

