function [] = ifftshow(f)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
f1=abs(f);
fm=max(f1(:));
figure,imshow(im2uint8(f1/fm));
end

