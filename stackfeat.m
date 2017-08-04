function [y] =stackfeat(x,N)
% stack N columns in the feature matrix x to form a new matrix y that contain these stacked vectors
% dpkbaby

[D,T]=size(x);
numwindows=T-N+1;
y=zeros(N*D,numwindows);

start_index=1;
for wndw=1:numwindows
    y(:,wndw)=reshape(x(:,start_index:start_index+N-1),[],1);
    start_index=start_index+1;
end
end
