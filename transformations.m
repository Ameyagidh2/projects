%image transformations
%thresholding 
i=imread('cameraman.tif');
%figure,imshow(i);
complement=imcomplement(i);
%figure,imshow(complement)
dark=imsubtract(i,50);
%figure,imshow(dark)
white=imadd(i,200);
%figure,imshow(white)
mult=immultiply(i,5);
%figure,imshow(mult)
div=imdivide(i,2);
%figure,imshow(div)
%or convert first into binary(>120) then double then again after using nos
%convert to uint8 format
%log and exponential transform
%x is log and y is  exponential
i1=imread('ameya_gidh.jpg');
%figure,imshow(i1);title('original_image')
im=im2double(i1);
x=im;%set x as new image and use i values initialially then modify them accordingly
[r,c]=size(im);
factor_c=1;
i=1;
j=1;
for i=1:r
    for j=1:c
        x(i,j)=factor_c*log(1+im(i,j));
        y(i,j)=factor_c*im(i,j)^2;
    end
end
%subplot(1,2,1);imshow(x);title('New image Logarithmic')
%subplot(1,2,2);imshow(y);title('New image Exponential')
%log makes brighter images reduce contrast and exponential makes bright
%images increase contrast
ameya=imread('ameya_gidh.jpg');
%ameya=imcomplement(ameya);
%imshow(ameya);
%Histogram and contrast stretching
histogram_image=imread('pout.tif');
%subplot(1,2,1);imshow(histogram_image);title('original image pout');
%subplot(1,2,2);imhist(histogram_image);

%imh=imadjust(histogram_image,[0.3,0.6],[0.0,1.0]);%my pixels are from 75/255 to 150/255 ie 0.3 to 0.6  intensity levels
%subplot(1,2,1);imshow(imh);title('adjusted histogram equalization')
%subplot(1,2,2);imhist(imh);
%imh1=histeq(histogram_image);
%subplot(1,2,1);imshow(imh1);title('Inbult histogram equalization')
%subplot(1,2,2);imhist(imh1);
close all
%high frequency components blur image at boundaries
%low frequency components lead to blurring of image
%LOW Pass Filter 1/9 on image ameya which is colored and is just showing black dots and cameraman
camera_man=imread('cameraman.tif');
%Using own filter
%filter_low=ones(5,5)/9;%change window size more window size more is the blurring
%low_pass_filter_image=filter2(filter_low,camera_man,'same');
%imshow(low_pass_filter_image/255)

%usinf bui;lt in function for applying filters
%filter_low_builtin=fspecial('average',[4,4]);%window size
%built_in=filter2(filter_low,camera_man,'full');
%imshow(built_in/255)

%Histogram Specification or histogram mapping
%matching 2 images

img_src=imread('flowers.tif');
ref_img=imread('flowers._spec.tif');
%figure;imshow(img_src);title('Image source');
%figure;imshow(ref_img);title('reference image');

%Seperate the reference image into corresponding channels of rgb values and planes
img_scr_r=img_src(:,:,1);
img_scr_g=img_src(:,:,2);
img_scr_b=img_src(:,:,3);


ref_img_r=ref_img(:,:,1);
ref_img_g=ref_img(:,:,2);
ref_img_b=ref_img(:,:,3);

% make these above values as histograms
h_ref_img_r=imhist(ref_img_r);
h_ref_img_g=imhist(ref_img_g);
h_ref_img_b=imhist(ref_img_b);
%after extracting the colour channels put these in the histeq function to
%map the rgb values of the two histograms

out_scr_r=histeq(img_scr_r,h_ref_img_r);
out_scr_g=histeq(img_scr_g,h_ref_img_g);
out_scr_b=histeq(img_scr_b,h_ref_img_b);

%now making the output histogram

histsp(:,:,1)=out_scr_r;
histsp(:,:,2)=out_scr_g;
histsp(:,:,3)=out_scr_b;

%subplot(2,2,1);imshow(img_src);title('Image source');
%subplot(222);imshow(ref_img);title('reference image');
%subplot(224);imshow(histsp);title('New result image after mapping the source and result histograms')
%method first extract the rgb color channels of reference image make the
%histogram from these using imhist function and map them using histeq
%function

%Gaussian noise
camera_man=imread('cameraman.tif');
%adding noise to image
noise=imnoise(camera_man,'gaussian',0.01);%variance
%subplot(221);imshow(camera_man);title('original image of camera man')
%subplot(223);imshow(noise);title('cameraman with noise')
sigma=3;
cutoff=3*sigma;
gaussian_filter=fspecial('gaussian',2*cutoff+1,sigma);
%generally donot use gaussian filter for gaussian noise
out=conv2(noise,gaussian_filter,'same');%same is size convolve the filter with image or noise
%subplot(224);imshow(out/256);title('camera man with noise')
% use of wiener filter for gaussian noise
w=wiener2(noise,[5,5]);
%subplot(222),imshow(w);title('camera man using wiener filter')

%Salt and pepper noise using median filter
salt_pepper_noise=imnoise(camera_man,'salt & pepper',0.01);
%subplot(2,2,1);imshow(camera_man);title('original image');
%subplot(2,2,2);imshow(salt_pepper_noise);title('Salt and peper noise added');
med=medfilt2(salt_pepper_noise);
%subplot(2,2,3);imshow(med);title('median filter applied for salt and pepper noise');

%order filtering
ord=ordfilt2(salt_pepper_noise,25,ones(10,10));%25th rank
%imshow(ord)





