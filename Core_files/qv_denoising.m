function [Iout]=qv_denoising(I,iter,dt,lambda, gamma)
% Input: I    - image (double array gray level 1-256),
%  iter - num of iterations,
%  dt   - time step [0.2],
%  ep   - epsilon (of gradient regularization) [1],
%  lambda  - fidelity term lambda [0],
%  I0   - input (noisy) image [I0=I]
%  Iout: evolved image

I0=I;
[ny,nx]=size(I);
for i=1:iter,  %% do iterations
    % estimate derivatives
    I_x = (I(:,[2:nx nx])-I(:,[1 1:nx-1]))/2;
    I_y = (I([2:ny ny],:)-I([1 1:ny-1],:))/2;
    I_xx = I(:,[2:nx nx])+I(:,[1 1:nx-1])-2*I;
    I_yy = I([2:ny ny],:)+I([1 1:ny-1],:)-2*I;    
    
    % gradient decent
    I_t = lambda*(2*I_xx+2*I_yy)-gamma.*(I-I0);
    I=I+dt*I_t;  % evolve image by dt
end
% return image
Iout=I;
