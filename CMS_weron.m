% CMS_weron - Simulate Stable random variables with 
%             Weron (beta_2, sigma_2) parametrisation 
%             and Weron choice for k(alpha)
%
% INPUT:
% a - alpha
% s - sigma2
% b - beta2
% m - mu
% n - number of stable random variables 
%
% OUTPUT:
% x - vector 1xn of stable iid random variables 
% 
% EXAMPLE:
% x = CMS_weron(1.2, 1, 0.5, 0, 100)


function x = CMS_weron( a, s, b, m , n ) 

% set seed for repeatable results
% rng(3); 

w = exprnd(1, n, 1); 
u = unifrnd(-1/2,1/2, n, 1); 
% inline functions 
eab = @(al,bl) (bl .* pi/2 .* al) .*(0<al & al<1) + ( bl .* pi/2 .* al - bl.*pi ) .* (1<al & al<2);  
tab = @(yl,al,bl) sin( pi.* al .*yl + eab(al,bl)) ./ ((cos( pi* yl )).^(1./al)) .* (cos( pi.* (al-1).*yl + eab(al,bl)) ).^((1-al)./al);  

if a~=1 
    z = tab(u,a,b).*w.^((a-1)/a); 
    x = s*z+m;  
else 
    z = (pi/2 + b*pi*u).* tan(pi*u) - b*log((w.*cos(pi*u))/(pi/2+b*pi*u));
    x = s*z + 2/pi*b*s*log(s) + m; 
end






