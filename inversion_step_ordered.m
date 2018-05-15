% inversion_step_ordered - invert v to y, given a couple (alpha, beta) 
%                          that have been proposed in the Metropolis step 
%
% INPUT:
% a  - alpha
% b  - beta2
% vI - set of 'fixed' v_i, sorted
% I  - index indicating the sorting order
% n  - lenght of the latent variables (is the same of the \alpha-stable dataset)
% 
% OUTPUT:
% y = tab^{-1}(v): (new) latent variables y corresponding to (old)v
%                  the output is in the same order as the original variables.
% 
% NOTES:
% the v-values in input are supposed to be pre-sorted in increasing order
% 
% EXAMPLE: oss here lab(1.2, 0.5) = 0.1667
% y = inversion_step_ordered(1.2, 0.5, [2, 3, 8], [3,2,1], 3) 

function y = inversion_step_ordered(a, b, vI, I, n) 

y=nan(n,1);   % output: not ordered (same order as z)
yI=nan(n,1);  % partial output: ordered 
tab_newton = cell(n,1);

eab = @(al,bl) ((bl .* pi/2 .* al) .*(0<al && al<1) + ( bl .* pi/2 .* al - bl.*pi ) .* (1<al && al<2));  
tab = @(yl,al,bl) (sin( pi.* al .*yl + eab(al,bl)) ./ ((cos( pi* yl )).^(1./al)) .* (cos( pi.* (al-1).*yl + eab(al,bl)) ).^((1-al)./al));  
lab = @(al,bl) (-eab(al,bl)./(pi.*al));

for j=1:n
    tab_newton{j,1} = @(yl)(tab(yl,a, b) - vI(j));
end
    
% tab derivative as a function only of y
tab_der_newton = @(yl) (a * pi * (cos(pi*yl)).^((a-1)/a) .* (cos( (a-1)*pi*yl + eab(a,b))).^(1/a) .*...
                       (1 + 1/(a^2) .* (tan(pi*yl) - (a-1).* tan((a-1)*pi*yl + eab(a,b))).^2) );
 
vIn = vI(vI<=0); Ln=length(vIn); %negative values to invert  
vIp = vI(vI>0);  Lp=length(vIp); %positive values to invert                     

% negative values 
if Ln~=0
    [yI(1)] = safe_newton(tab_newton{1,1}, tab_der_newton, -1/2, lab(a,b), 10^(-10), 10^(-10), 10^3); 
    for j=2:Ln
        [yI(j)] = safe_newton(tab_newton{j,1}, tab_der_newton, yI(j-1)-eps, lab(a,b), 10^(-10), 10^(-10), 10^3); 
    end
end

% positive values 
if Lp~=0
    [yI(Ln+1)] = safe_newton(tab_newton{Ln+1,1}, tab_der_newton, lab(a,b), 1/2, 10^(-10), 10^(-10), 10^3); 
    for j=2:Lp
        [yI(Ln+j)] = safe_newton(tab_newton{Ln+j,1}, tab_der_newton, yI(Ln+j-1)-eps, 1/2, 10^(-10), 10^(-10), 10^3); 
    end
end

% re-ordering of the y, according to z (data) 
y(I)=yI;
 


