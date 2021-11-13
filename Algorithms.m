%Bitplane slicing of images used for compression of images
im1=imread('cameraman.tif');
%figure,imshow(im1)
%Convert to double form
im2=double(im1);
%figure,imshow(im2/255)%double image to uint8 form to display
%you can use im2=im2double(im1)
%a=im1>120;
%figure,imshow(a)
%impixelinfo(a)

%Now for bitplane slicing extract MSB,LSB AND MIDDLE Part of the image
c1=mod(floor(im2/2),2);%double image as input
%figure,imshow(c1)
%Breaking of image which is in double form into small bitplane sliced parts
c0=mod(floor(im2/2),2);
%figure,imshow(c0)
c2=mod(floor(im2/4),2);
%figure,imshow(c2)
c3=mod(floor(im2/8),2);
%figure,imshow(c3)
c4=mod(floor(im2/16),2);
%figure,imshow(c4)
c5=mod(floor(im2/32),2);
%figure,imshow(c5)
c6=mod(floor(im2/64),2);
%figure,imshow(c6)
c7=mod(floor(im2/128),2);
%figure,imshow(c)
%Forming the combined image
%ccnet_combined=2*(2*(2*(2*(2*(2*(2*c7+c6)+c5)+c4)+c3)+c2)+c1)+c0;
%figure,imshow(uint8(ccnet_combined))

%Resizing image
%Inbuild functons
%resize1=imresize(im1,1/2);
%figure,imshow(resize1)
%figure,imshow(im1)

%Resizing image using own algorithm i.e. using the way this is done inside
%the algorithm
[rows,columns]=size(im1);
i=1;j=1;
%leave alternate pixels aside and make an image of smaller size
%c is a new matrix of half size and make initially of zeros then increase
%its size eventually
c=zeros(rows/2,columns/2);
for x =1:2:rows
    for y=1:2:columns
      c(i,j)=im1(x,y);
      j=j+1;
    end 
i=i+1;%we need i and j from c to move in consecutive fashion not like x nd y having a difference of 2
j=1;
end
%figure,imshow(c/255)
%figure,imagesc(c),colormap(gray)





