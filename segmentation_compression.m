%water marking for encryption and hiding your image information

%image compression dct
ameya_gidh=imread('ameya_gidh.jpg');
%imshow(ameya_gidh)
gray_ameya=rgb2gray(ameya_gidh);%Use gray scale images instead of rgb types for dct transform
%imshow(gray_ameya)
g1=dct2(gray_ameya);
%see how the pixels are in the form of an image so first handle for image
%needed
%g2=imagesc(g1);
%impixelregion(g2);

%Image compression code 
%Gray image compression
%I = imread('cameraman.tif');
%I = im2double(I);
%T = dctmtx(8);
%dct = @(block_struct) T * block_struct.data * T';
%B = blockproc(I,[8 8],dct);
%invdct = @(block_struct) T' * block_struct.data * T;
%mask = [1   1   1   1   0   0   0   0
 %       1   1   1   0   0   0   0   0
  %      1   1   0   0   0   0   0   0
   %     1   0   0   0   0   0   0   0
    %    0   0   0   0   0   0   0   0
     %   0   0   0   0   0   0   0   0
       % 0   0   0   0   0   0   0   0
      %  0   0   0   0   0   0   0   0];
%B2 = blockproc(B,[8 8],@(block_struct) mask .* block_struct.data);
%I2 = blockproc(B2,[8 8],invdct);
%subplot(2,2,1);imshow(I);title('original image');
%subplot(2,2,3);imshow(I2);title('compressed image')

%RGB Image compression
ameya=imread('ameya_gidh.jpg');
%subplot(2,2,1);imshow(ameya);title('Original image')
R_C=im2double(ameya(:,:,1));
G_C=im2double(ameya(:,:,2));
B_C=im2double(ameya(:,:,3));
%Extract the RGB Channels

%subplot(2,2,2);imshow(R_C);title('Red channels');
%subplot(2,2,3);imshow(G_C);title('Green channels');
%subplot(2,2,4);imshow(B_C);title('Blue channels');

%divide  the image into 8x8 blocks using blockproc function and apply dct
T=dctmtx(8);
dct=@(block_struct)T*block_struct.data*T';
R_C_B=blockproc(R_C,[8 8],dct);
G_C_B=blockproc(G_C,[8 8],dct);
B_C_B=blockproc(B_C,[8 8],dct);
mask=[1 1 1 1 0 0 0 0
      1 1 1 0 0 0 0 0 
      1 1 0 0 0 0 0 0
      0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0];%change mask for changing compression ratio of image and blurrness also more 1s more clear  the image
  %applying the mask here which causes blurring effect and reduce quality
R_C_B2 = blockproc(R_C_B,[8,8],@(block_struct)mask.*block_struct.data);
G_C_B2 = blockproc(G_C_B,[8,8],@(block_struct)mask.*block_struct.data);
B_C_B2 = blockproc(B_C_B,[8,8],@(block_struct)mask.*block_struct.data);

%inverse dct
invdct=@(block_struct)T'*block_struct.data*T;
R_12=blockproc(R_C_B2,[8 8],invdct);
G_12=blockproc(G_C_B2,[8 8],invdct);
B_12=blockproc(B_C_B2,[8 8],invdct);
compress=cat(3,R_12,G_12,B_12);
%imwrite(ameya,'original.jpg')
%imwrite(compress,'compressed_ameya_image.jpg');
%imshowpair(ameya,compress,'montage')

%Image segmentation and getting certain parts from an entire image
%using threshold value adaptive and otsu method
rice=imread('rice.tif');
%subplot(2,2,1);imshow(rice);title("original image");
%whos r
%imfinfo('rice.tif');
mri=imread('mri.tif');
%subplot(2,2,1),imshow(mri);title('actual image');
T1=28;%from histogram plotting
T2=graythresh(rice);
%figure;imhist(rice);
%as we can clearly seperate the background and the foreground you can use
%thresholding or give a value you like else go for adaptive thresholding.
rice1=im2bw(rice,T1/255);
rice2=im2bw(rice,T2/255);
%figure,imshow(rice1);
T3=blkproc(mri,[15,15],@adapt);
%subplot(2,2,2);imshow(T3);title('seperated image')

%image labelling after segmentation

coins=imread('coins.tif');
%subplot(2,2,1);imshow(a);title('coins image');
b=im2bw(coins);
%subplot(2,2,2);imshow(b);title('Black and white image');
filled=imfill(b,'holes');
%subplot(2,2,3);imshow(filled);title('holes filled image');
%label individual images
label=bwlabel(filled);
m=max(max(label));%length of all objects in the given image
%finding objects in image
im1=(label==1);
%imshow(im1);

for j=1:m
    [row,col]=find(label==j);%rows and columns in the label
    len=max(row)-min(row)+2;
    breadth=max(col)-min(col)+2;
    target=uint8(zeros([len breadth]));
    %iterations in x and y from sx and sy
    sy=min(col)-1;
    sx=min(row)-1;
  %assigning the labels to target values  
for i=1:size(row,1)
    % x and y are rows of target image
    x=row(i,1)-sx;
    y=col(i,1)-sy;
    target(x,y)=coins(row(i,1),col(i,1));
end
mytitle=strcat('object Number',num2str(j));
figure,imshow(target);title(mytitle);
end
    
    
    
    
    
    







