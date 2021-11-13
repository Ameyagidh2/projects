function [y] = adapt(image)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if std2(image)<1
    y=ones(size(image,1),size(image,2));
    %background then directly classify using std value
else 
    y=im2bw(image,graythresh(image));

end

