function [Omega,OPara] = DictLineSearch03_analysis_rowOperation(X_sparse,Omega, Signal, IPara)
% DictLineSearch03_analysis_rowOperation is the dictionary update function in Analysis SimCO
% This file is modified form the "DictUpdate03.m" in the code of SimCO.
% @ September 2013, University of Surrey

% Details about the algorithm can be found in the paper:
% J. Dong, W. Wang, W. Dai, M. D. Plumbley, Z. Han, J. Chambers
% "Analysis SimCO Algorithms for Sparse Analysis Model Based Dictionary Learning,"
% submitted to IEEE Transactions on Signal Processing, Feburary 2015.
%
% The codes & data have been deposited to http://dx.doi.org/10.15126/surreydata.00808101.
%
% Input parameters:-
% X_Sparse: sparse coefficients
% Omega: initial dictionary
% Signal: training data/signals
% IPara: parameter sets (defined below)
%
% Output parameters:-
% Omega: updated dictionary
% OPara: parameter sets (defined below)
%
% Written by Jing Dong, moderated by Wenwu Wang, version 1.0                                    
%
% If you have any questions or comments regarding this package, or if you want to 
% report any bugs or unexpected error messages, please send an e-mail to
% w.wang@surrey.ac.uk
%         
% Copyright 2015 J. Dong, W. Wang, W. Dai, M. D. Plumbley, Z. Han, and J. Chambers
% 
% This software is a free software distributed under the terms of the GNU 
% Public License version 3 (http://www.gnu.org/licenses/gpl.txt). You can 
% redistribute it and/or modify it under the terms of this licence, for 
% personal and non-commercial use and research purpose.

% Additional reference:
% W. Dai, T. Xu, and W. Wang, 
% "Simultaneous Codeword Optimization (SimCO) for Dictionary Update and Learning,"
% submitted to IEEE Transactions on Signal Processing, October 2011.
% Full text is available at http://arxiv.org/abs/1109.5302
% Wei Dai, Tao Xu, Wenwu Wang
% Imperial College London, University of Surrey
% wei.dai1@imperial.ac.uk  t.xu@surrey.ac.uk  w.wang@surrey.ac.uk
% October 2011


% IPara.gmin : if the gradient frobenius norm is less than gmin, the
%   gradient can be viewed as zero. default value is 1e-5.
% IPara.Lmin : t4-t1 should be larger than Lmin. default value is 1e-6.
% IPara.rNmax : the number of iterative refinement in Part B. default value
%   is 3
% IPara.t4 : the initial value of t4. default value is 1e-2.
% IPara.mu : regularization parameter

% OPara.f0 : initial value of f
% OPara.f1 : final value of f
% OPara.fv : track the value of f
% OPara.tv : track the value of t
% OPara.topt : optimal t value
% OPara.gn2 : the frobenius norm of the gradient
% OPara.Flag = 1 : gradient is too small
% OPara.Flag = 0 : successful update
% OPara.Flag = -1 : t is too small


%% initialization
c = (sqrt(5)-1)/2;
fv = zeros(1,100);
tv = zeros(1,100);
f4v = zeros(1,4);
t4v = zeros(1,4);
% [m,n] = size(Y);
% d = size(D,2);
[m,n] = size(X_sparse);
d = size(Omega,2);

if nargin >= 4 && isfield(IPara,'gmin')
    gmin = IPara.gmin;
else 
    gmin = 1e-5;
end
if nargin >= 4 && isfield(IPara,'Lmin')
    Lmin = IPara.Lmin; 
else
    Lmin = 1e-6;
end
if nargin >= 4 && isfield(IPara,'rNmax')
    rNmax = IPara.rNmax;
else
    rNmax = 3;
end
if nargin >= 4 && isfield(IPara,'t4') 
    t4v(4) = IPara.t4;
else
    t4v(4) = Lmin/(1-c);
end

%% compute the direction and corresponding gradient
% [f,g] = fg_tilde_eval01_analysis(Y,D,Signal);  % jing
[f,g] = fg_tilde_eval01_analysis_rowOperation(X_sparse,Omega,Signal);  % jing
% [f,g] = fg_tilde_eval01_analysis(Y,D',Signal);  % jing
evaln = 1;
fv(evaln) = f;
tv(evaln) = 0;

OPara.gn2 = norm(g,'fro'); % jing
gColn2 = sqrt(sum(g.*g,2));
gZero = gColn2 < gmin*norm(X_sparse,'fro')^2/n;

%if the the magnitude of the gradient is less than the minimum threshold 
%   value, then quit and return D and X
if sum( gZero ) == size(Omega,2) 
    OPara.Flag = 1;
    OPara.fv = fv(1:evaln);
    OPara.tv = tv(1:evaln);
    OPara.topt = 0;
    OPara.f0 = f;
    OPara.f1 = f;
    return;
end

gColn2(gZero) = 0;
H = zeros(m,d);
% H(1,gZero) = 1;
% H(2:m,gZero) = 0;
H(gZero, 1) = 1;
H(gZero, 2:d) = 0;
% H(:,~gZero) = g(:,~gZero).*repmat(-1./gColn2(~gZero),m,1);
H(~gZero, :) = g(~gZero, :).*repmat(-1./gColn2(~gZero),1,d);
Step = gColn2/mean(gColn2);

%% Part A : find a good t4
% set t4v and f4v; 
t4v(3) = t4v(4)*c; t4v(2) = t4v(4)*(1-c);
f4v(1) = fv(1);
for evaln = 2:4
    t = t4v(evaln);
%     Dt = D.*repmat(cos(Step*t),m,1) + H.*repmat(sin(Step*t),m,1); 
%     f4v(evaln) = fg_tilde_eval01_analysis(Y,Dt,Signal); 
    Omegat = Omega.*repmat(cos(Step*t),1,d) + H.*repmat(sin(Step*t),1, d); % jing
    f4v(evaln) = fg_tilde_eval01_analysis_rowOperation(X_sparse,Omegat,Signal); % jing
end
fv(2:4) = f4v(2:4);
tv(2:4) = t4v(2:4);
% loop to find a good t4
while t4v(4)-t4v(1) >= Lmin
    % if f(D(t1)) is not greater than f(D(t2)), then t4=t2, t3=c*t4,
    %   t2=(1-c)*t4
    if f4v(1)<=f4v(2)
        t4v(4) = t4v(2); t4v(3) = t4v(4)*c; t4v(2) = t4v(4)*(1-c);
        f4v(4) = f4v(2);
        evaln = evaln + 1;
        t = t4v(2); tv(evaln) = t;
        %         Dt = D.*repmat(cos(Step*t),m,1) + H.*repmat(sin(Step*t),m,1);
        %         ft = fg_tilde_eval01_analysis(Y,Dt,Signal); % jing
        Omegat = Omega.*repmat(cos(Step*t),1,d) + H.*repmat(sin(Step*t),1, d); % jing
        ft = fg_tilde_eval01_analysis_rowOperation(X_sparse,Omegat,Signal); % jing
        f4v(2)=ft; fv(evaln)=ft;
        evaln = evaln + 1;
        t = t4v(3); tv(evaln) = t;
%         Dt = D.*repmat(cos(Step*t),m,1) + H.*repmat(sin(Step*t),m,1);
%         ft = fg_tilde_eval01_analysis(Y,Dt,Signal);% jing
        Omegat = Omega.*repmat(cos(Step*t),1,d) + H.*repmat(sin(Step*t),1, d); % jing
        ft = fg_tilde_eval01_analysis_rowOperation(X_sparse,Omegat,Signal); % jing
        f4v(3) = ft; fv(evaln) = ft;
        % if f(D(t2)) is not greater than f(D(t3)), then t4=t3, t3=t2,
        %   t2=(1-c)*t4 
    elseif f4v(2)<=f4v(3)
        t4v(4) = t4v(3); t4v(3) = t4v(2); t4v(2) = t4v(4)*(1-c);
        f4v(4) = f4v(3); f4v(3) = f4v(2);
        evaln = evaln + 1;
        t = t4v(2); tv(evaln) = t;
%         Dt = D.*repmat(cos(Step*t),m,1) + H.*repmat(sin(Step*t),m,1);
%         ft = fg_tilde_eval01_analysis(Y,Dt,Signal);% jing
        Omegat = Omega.*repmat(cos(Step*t),1,d) + H.*repmat(sin(Step*t),1, d); % jing
        ft = fg_tilde_eval01_analysis_rowOperation(X_sparse,Omegat,Signal); % jing

        f4v(2) = ft; fv(evaln) = ft;
        % if f(D(t3)) is greater than f(D(t4)), then t2=t3, t3=t4
        %   t4=t3/c 
    elseif f4v(3)>f4v(4)
        t4v(2) = t4v(3); t4v(3) = t4v(4); t4v(4) = t4v(3)/c;
        f4v(2) = f4v(3); f4v(3) = f4v(4); 
        evaln = evaln + 1;
        t = t4v(4); tv(evaln) = t;
%         Dt = D.*repmat(cos(Step*t),m,1) + H.*repmat(sin(Step*t),m,1);
%         ft = fg_tilde_eval01_analysis(Y,Dt,Signal);% jing
        Omegat = Omega.*repmat(cos(Step*t),1,d) + H.*repmat(sin(Step*t),1, d); % jing
        ft = fg_tilde_eval01_analysis_rowOperation(X_sparse,Omegat,Signal); % jing

        f4v(4) = ft; fv(evaln) = ft;
    else
        %quit
        break;
    end
end
% if t is too small, fet Flag to minus 1
if t4v(4)-t4v(1) < Lmin
    Flag = -1;
end

%% Part B: refine the segment
evalN = evaln;
% iterate until t4-t1 is small enough
while (t4v(4)-t4v(1)) >= Lmin && evaln - evalN <= rNmax
    % if f(D(t1))>f(D(t2))>f(D(t3)), then t1=t2, t2=t3, t3=t1+c*(t4-t1)
    if f4v(1)>f4v(2) && f4v(2)>f4v(3)
        t4v(1) = t4v(2); t4v(2) = t4v(3); t4v(3) = t4v(1)+c*(t4v(4)-t4v(1));
        f4v(1) = f4v(2); f4v(2) = f4v(3);
        evaln = evaln + 1;
        t = t4v(3); tv(evaln) = t;
%         Dt = D.*repmat(cos(Step*t),m,1) + H.*repmat(sin(Step*t),m,1);
%         ft = fg_tilde_eval01_analysis(Y,Dt,Signal);% jing
        Omegat = Omega.*repmat(cos(Step*t),1,d) + H.*repmat(sin(Step*t),1, d); % jing
        ft = fg_tilde_eval01_analysis_rowOperation(X_sparse,Omegat,Signal); % jing
        f4v(3) = ft; fv(evaln) = ft;
    % otherwise, t4=43, t3=t2, t2=t1+(1-c)(t4-t1)    
    else
        t4v(4) = t4v(3); t4v(3) = t4v(2); t4v(2) = t4v(1)+(1-c)*(t4v(4)-t4v(1));
        f4v(4) = f4v(3); f4v(3) = f4v(2);
        evaln = evaln + 1;
        t = t4v(2); tv(evaln) = t;
%         Dt = D.*repmat(cos(Step*t),m,1) + H.*repmat(sin(Step*t),m,1);
%         ft = fg_tilde_eval01_analysis(Y,Dt,Signal); % jing
        Omegat = Omega.*repmat(cos(Step*t),1,d) + H.*repmat(sin(Step*t),1, d); % jing
        ft = fg_tilde_eval01_analysis_rowOperation(X_sparse,Omegat,Signal); % jing
        f4v(2) = ft; fv(evaln) = ft;
    end
    0;
end

%% finalize
fv = fv(1:evaln);
tv = tv(1:evaln);
[fmin,findex] = min(fv);
t = tv(findex);

% D = D.*repmat(cos(Step*t),m,1) + H.*repmat(sin(Step*t),m,1);
Omega = Omega.*repmat(cos(Step*t),1,d) + H.*repmat(sin(Step*t),1, d); % jing
Omega = normr(Omega);
%jing
f = fg_tilde_eval01_analysis_rowOperation(X_sparse,Omega,Signal); 
% [f,X] = fg_tilde_eval01(Y,D,Omega,IPara);

OPara.f0 = fv(1);
OPara.f1 = f;
OPara.fv = fv;
OPara.tv = tv;
OPara.topt = t;
OPara.Flag = 0;
