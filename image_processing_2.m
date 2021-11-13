%High pass filter
%first order derivative
circuit=imread('circuit.tif');
%subplot(2,2,1);imshow(circuit);title('Circuit');
%edge detection
e1=edge(circuit,'prewitt');
%subplot(2,2,2);imshow(e1);title('prewitt');
e2=edge(circuit,'canny');
%subplot(2,2,3);imshow(e2);title('canny');
e3=edge(circuit,'sobel');
%subplot(2,2,4);imshow(e3);title('sobel');
e4=edge(circuit,'roberts');
%subplot(2,2,1);imshow(e4);title('roberts');

% making a filter
px=[-1,0,1;-1,0,1;-1,0,1];
py=px;
ax=filter2(px,circuit);
ay=filter2(py,circuit);
pedge=sqrt(ax.^2+ay.^2);
%imshow(pedge/255)

%2nd order derivative  high pass filter laplacian
camera_man=imread('cameraman.tif');
%lap=fspecial('laplacian')%great results
%lap_filter_image=filter2(lap,camera_man);
%imshow(lap_filter_image);
%own image filter laplacian
h=[0 1 0;1,-4,1;0 1 0];
im1=conv2(h,camera_man);
%imshow(im1

e=edge(camera_man,'log');
%figure,imshow(e);
disk=fspecial('disk');
img=edge(camera_man,'zerocross',disk);
%imshow(img);

%frequency domain fourier transform
%perform filter operations in frequency domain as these operations are less computationally heavy compared to conv2
x=1:0.1:10;
y=sin(x);
%plot(x,y)
%fft and ifft works on vectors and fft2 and iffft2 works on matrices
x=[2,3,4];
y=fft(x);
b=ifft(y);
%making a black box
a=zeros(256);
%imshow(a);
a(78:178,78:178)=1;
%imshow(a)%black box with a white square
%zero is black and 255 is white
%Transforming and changing the a box in the frequency domain to apply
%filters as convolve in spatial domain is computationally heavy
af=fftshift(fft2(a));%shift to center use fftshift command
%imshow(af);
%applying filter using log function find max and divide the function or
%directly use mat2gray inbuilt function
af1=log(1+abs(af));
%imshow(af1)
fm=max(af1(:));
af2=af1/fm;
%imshow(im2uint8(af2));
%or inbuilt method
%imshow(mat2gray(log(1+abs(af))))

%Ringing effect using a circle
[x,y]=meshgrid(-128:127,-128:127);
%imshow(x);
z=sqrt(x.^2+y.^2);
c=z<15;%white circle of cutoff radius of 15
%imshow(c)
%ringing effect
af3=fftshift(fft2(c));
%imshow(mat2gray(log(1+abs(af3))));




