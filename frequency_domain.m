%Ringing effect present which can be removed using butterworth filter
camera_man=imread('cameraman.tif');
[x,y]=meshgrid(-128:127,-128:127);
z=sqrt(x.^2+y.^2);
cl=z<15;% low pass filter
%convert in frequency domain and shift
af=fftshift(fft2(camera_man));
%imshow(af);
%apply filter on image
aff1=af.*cl;
%fftshow(af1);
aff2=ifft2(aff1);
%ifftshow(af2);%back to spacial domain

% High pass filter 
ch=z>15;
hpf=af.*ch;%fft shift of camera_man image
%fftshow(hpf);
hf1=ifft(hpf);
%ifftshow(hf1);%back to spacial domain


%Butterworth filter to remove ringing effect
a=imread('cameraman.tif');
%figure,imshow(a);
bf=butterhp(a,15,1);%change and tune these values
afg=fftshift(fft2(a));
%fftshow(afg);
aff1=afg.*bf;
%fftshow(aff1);
aff2=ifft2(aff1);
%ifftshow(aff2);

%Gaussian filter
gau_filter=fspecial('gaussian',256,10);
max(gau_filter(:));%shift to center
g1=mat2gray(gau_filter);
max(g1(:));

af=fftshift(fft2(camera_man));
%fftshow(af)
ag1=af.*g1;
ag2=ifft2(ag1);
%ifftshow(ag2)

%improve illumination in image use homomorphic filter
a=imread('trees.tif');
im=im2double(a);
i=fft2(log(im+0.01));
%use any high pass filter now
f=butterhp(a,15,1);
c=i.*f;
h=real(ifft2(c));
h1=exp(h);
%ifftshow(h1);











