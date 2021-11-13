%grey scale
im1=imread('cameraman.tif');
%imshow(im1)
%im2=imread('circbw.tif');
%subplot(224);imshow(im2)
subplot(221);imshow(im1)
%figure
%im3=imread('eight.tif');
%subplot(222);imshow(im3)
%size(im1)
%imfinfo('cameraman.tif')
colormap('jet')
%colormap('spring')
impixelinfo
%(157, 134)  169
black_img=uint8(zeros(512,512));
subplot(2,2,2);imshow(black_img)
white_img=uint8(255*ones(512,512));
subplot(2,2,3);imshow(white_img)
