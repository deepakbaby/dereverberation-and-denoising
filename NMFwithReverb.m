function [H, gX,cost]=NMFwithReverb(Z_stacked, A, windowlength,lambda, numspeechexemplars, numiter,L, updateH, computecost)

% REQUIRES GPU (Comment out the GPU Array lines for converting it to the CPU version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% If you use this code please cite
%
% [1] Deepak Baby and Hugo Van hamme. Supervised Speech Dereverberation in 
% Noisy Environments using Exemplar-based Sparse Representations. 
% In Acoustics, Speech and Signal Processing (ICASSP), 2016 IEEE 
% International Conference on, Shanghai, China, March 2016. 
%
% [2] N. Mohammadiha, P. Smaragdis, and S. Doclo. Joint acoustic and spectral modeling for speech dereverberation using non-
% negative representations, in Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on,
% April 2015, pp. 4410-4414.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code to estimate the activations and RIR using the NMF-based formulation
% in the paper [1].

% Inputs:
%   Z_stacked : stacked input data matrix (reverberated)
%   A : Dictionary Matrix ; contains speech and noise dictionaries A=[S N]
%   windowlength : number of frames stacked (variable T in paper [1])
%   L : length of the RIR to be estimated
%   lambda : sparsity penalty for speech and noise  ; lambda = [lambda_s lambda_n]
%   numspeechexemplars : number of speech exemplars
%   numiter : number of iterations
%   updateH : Is RIR to be estimated ? updateH=0 yields the traditional NMD activations
%   computecost : should the cost after every iteration be computed ?

% Outputs :
%   activations : output activations for exemplars
%   H : magnitude STFT of the RIR model
%   cost : cost after every iteration (for checking convergence)

% Written By Deepak Baby, KU Leuven, September 2015.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gY = gpuArray(single(Z_stacked));
A = gpuArray(single(A));
%---initialization for H(RIR)----
if updateH
    rng(666);
    h = rand(size(gY,1)/windowlength,L);
    gh = gpuArray(single(h));
    H = makeHfromh(gh,windowlength, alpha1);
else
    H = gpuArray.ones(size(gY,1),1);
    L=1;
end

[rows,~] = size(gY);
epsilon=1e-30; % to avoid 0/0
gepsilon=gpuArray(single(epsilon));


%---initialization for X(Activations)-----
gW_trans=A';
gX=gW_trans*gY;
cost = zeros(1,numiter);
if eq(size(lambda),[1 1])
    lambda=lambda*ones(size(A,2),size(gY,2));
elseif eq(size(lambda,2),1)
    lambda=repmat(lambda, 1, size(gY,2));
end

for iter=1:numiter
    
    Stilde_speech = A(:,1:numspeechexemplars)*gX(1:numspeechexemplars,:);
    %------updating Ytilde after getting a new value for H-----------
    gYtilde =  applyH_nmf(Stilde_speech,H,windowlength); % reverberated speech model
    gYtilde = gYtilde + A(:,numspeechexemplars+1:end)*gX(numspeechexemplars+1:end,:); % adding the noise reconstrction with the speech model
    gYtilde = max(gYtilde,1e-30);
    
    %-----------update for X------------------
    ratio = gY./gYtilde;
    ratio1 = [ratio zeros(size(gY,1),L-1)];
    
    %-----------update for Xs ------------------
    dummyones = gpuArray.ones(size(ratio1));
    fnss = @(ta1) bsxfun(@times,dummyones(:,ta1+1 : size(gY,2)+ta1), H(:,ta1+1));
    bb1 = arrayfun(fnss, 0:L-1, 'UniformOutput', false);
    summ1 = sum(cat(3, bb1{:}),3);
    denominator_Xs = A(:,1:numspeechexemplars)'*summ1;
    
    fn3 = @(ta) bsxfun(@times,ratio1(:,ta+1 : size(gY,2)+ta), H(:,ta+1));
    bb = arrayfun(fn3, 0:L-1, 'UniformOutput', false);
    summ = sum(cat(3, bb{:}),3);
    numerator_Xs = A(:,1:numspeechexemplars)'*summ;
    
    % -------- update for Xn ---------------------
    numerator_Xn = A(:,numspeechexemplars+1:end)'*ratio;
    denominator_Xn = A(:,numspeechexemplars+1:end)'*gpuArray.ones(size(ratio));
    
    numerator_X = [numerator_Xs ; numerator_Xn];
    denominator_X = [denominator_Xs; denominator_Xn];
    
    gX = gX.* max(numerator_X,1e-30)./(max(denominator_X,1e-30)+lambda);
    
    if updateH
        Stilde_speech = A(:,1:numspeechexemplars)*gX(1:numspeechexemplars,:);
        %------updating Ytilde after getting a new value for H-----------
        gYtilde =  applyH_nmf(Stilde_speech,H,windowlength); % reverberated speech model
        gYtilde = gYtilde + A(:,numspeechexemplars+1:end)*gX(numspeechexemplars+1:end,:); % adding the noise reconstrction with the speech model
        gYtilde = max(gYtilde,1e-30);
        
        ratio = gY./gYtilde;
        
        St= [zeros(size(Stilde_speech,1), L-1) Stilde_speech]; % zero padding to the left for right shifting
        fn1 = @(tau) sum(ratio.*St(:,L-tau:end-tau),2); % Summing over the columns for tau.
        fn2 = @(tau) sum(St(:,L-tau:end-tau),2);
        numerator_H =arrayfun(fn1, 0:L-1, 'UniformOutput', false);
        num_H=[numerator_H{:}];
        gnum_H = gpuArray(single(num_H));
        denominator_H = arrayfun(fn2, 0:L-1, 'UniformOutput', false);
        
        denom_H=[denominator_H{:}];
        gdenom_H = gpuArray(single(denom_H));
        gdenom_H = max(gdenom_H,1e-30);
        
        numer_frame = sum(permute(reshape(gnum_H',L,rows/windowlength,[]),[2,1,3]),3);
        denom_frame = sum(permute(reshape(gdenom_H',L,rows/windowlength,[]),[2,1,3]),3);
        
        
        gh = gh .* max(numer_frame,1e-30) ./ max(denom_frame,1e-30) ;
        H = makeHfromh(gh,windowlength);
        
    end
    
    if 0
        figure(1), imagesc(log(gX+1e-30)), colormap jet
        figure(2), imagesc(H(1:size(gh,1),:)), colormap jet, pause
    end
    if computecost
        cost1 = gY .* log((gY+gepsilon) ./(gYtilde + gepsilon)) - gY + gYtilde;
        cost2 = lambda .* gX;
        
        cost(iter) = sum(sum(cost1)) + sum(sum(cost2));
    end
    
end % numiters
end % EOF


function Z = applyH_nmf(Y,H,winlength)

% To reconstruct the reverberated spectrogram Z from non-reverberated spectrogram Y with RIR weights in H using the reverb formulation

[BT,L] = size(H);
B = BT/winlength;

feat = [zeros(B,winlength-1) reshape(gather(Y(:,1)),B,[])];
featstacked = gpuArray(single(stackfeat(feat,winlength)));
Y_new = [featstacked(:,1:end-1) Y];

fnrir = @(tau) bsxfun(@times, [zeros(BT,tau-1) Y_new(:,1:end-tau+1)], H(:,tau));
Z = arrayfun(fnrir, 1:L, 'UniformOutput', false); Z = cat(3, Z{:});
Z = sum(Z,3);
Z = Z(:,winlength:end);
end


function [H,h] = makeHfromh(h, windowlength)
% make H of size BTxL from h of size BxL
h = max(h,1e-30);
h = bsxfun(@rdivide,h,h(:,1));

for t=2:size(h,2)
    h (:,t) = min(h(:,t), h(:,t-1));
end
h = bsxfun(@rdivide, h, sum(h,2));
H = repmat(h, windowlength,1);
end
