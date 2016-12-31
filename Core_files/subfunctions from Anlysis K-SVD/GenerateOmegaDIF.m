function [Omega] = GenerateOmegaDIF(n)
%GenerateOmegaDIF generates the Omega_DIF analysis dictionary.
%  [Omega] = GenerateOmegaDIF(n)
%  ========================================================================
%  Input:
%  n - patch size (the atom dimension is d=n^2).
%  ========================================================================
%  Output:
%  Omega - analysis dictionary.
%  ========================================================================
%  Tomer Peleg
%  Department of Electrical Engineering
%  Technion, Haifa 32000 Israel
%  tomerfa@tx.technion.ac.il
%
%  October 2012
%  ========================================================================
Omega=zeros(2*n^2,n^2);
count=1;
for k=1:1:n
    for j=1:1:n-1,
        Image=zeros(n,n);
        Image(k,j)=1; 
        Image(k,j+1)=-1; 
        Omega(count,:)=Image(:)';
        count=count+1;
    end
    Image=zeros(n,n);
    Image(k,n)=1; 
    Image(k,1)=-1; 
    Omega(count,:)=Image(:)';
    count=count+1;
end
for k=1:1:n
    for j=1:1:n-1,
        Image=zeros(n,n);
        Image(j,k)=1; 
        Image(j+1,k)=-1; 
        Omega(count,:)=Image(:)';
        count=count+1;
    end
    Image=zeros(n,n);
    Image(n,k)=1; 
    Image(1,k)=-1; 
    Omega(count,:)=Image(:)';
    count=count+1;
end
Omega=Omega/sqrt(2); 