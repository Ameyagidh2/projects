aut=imread('autumn.tif');
%imfinfo('autumn.tif') 
%given image is rgb form
%imshow(aut)
%impixelinfo
%image pixel coordinates (147, 72)  [56 53 40]  #pixel values[56,53,40]
%impixel(aut,147, 72)
%no effect of color map on rgb image

%Indexed image is emu.tif
[b,bmap]=imread('emu.tif');
%imshow(b,bmap)
%These images save the sapce compared to rgb images in color type
%imfinfo('emu.tif')

%Image conversion from one form to another
%RGB TO GRAY
%aut_gray=rgb2gray(aut);
%figure,imshow(aut_gray);

%Indexed to gray scale
%ind_gray=ind2gray(b,bmap);
%figure,imshow(ind_gray);







