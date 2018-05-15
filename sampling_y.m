% sampling_y - adaptive rejection sampling of latent variables y, as
%              described in Buckle(1995). One latent variable is drawn from
%              the full conditional distribution \pi(y|\alpha, \beta, z_i)
%              by building a piecewise constant envelope on it. 
%
% INPUT:
% a  - alpha 
% b  - beta
% z  - standard alpha-stable sample 
% N  - sample size
%
% OUTPUT:
% y  - Nx1 vector containing one latent variable for each data point z_i 
%
% EXAMPLE:
% y = sampling_y(1.2 , 0.7,CMS_weron(1.2, 1, 0.7, 0, 100),100)


function y = sampling_y(a,b,z,N) 

% y preallocation: one latent variable y_i for each data point z_i  
y = nan(N,1);

% general function 
eab = @(al,bl) ((bl .* pi/2 .* al) .*(0<al && al<1) + ( bl .* pi/2 .* al - bl.*pi ) .* (1<al && al<2));  
tab = @(yl,al,bl) (sin( pi.* al .*yl + eab(al,bl)) ./ ((cos( pi* yl )).^(1./al)) .* (cos( pi.* (al-1).*yl + eab(al,bl)) ).^((1-al)./al));  
lab = @(al,bl) (-eab(al,bl)./(pi.*al));
f_yz = @(yl,zl,al,bl) ( al./abs(al-1).*  (exp( -(( abs( zl./tab(yl,al,bl) ) ).^(al./(al-1))) ) )  .* ((abs(zl./tab(yl,al,bl))).^(al./(al-1)))./abs(zl));

for i=1:N   
    accept = false;
 
    if z(i)>=0     
        jt = [lab(a,b),1/2]';  % initial jump points (grid domain)
    else
        jt = [-1/2,lab(a,b)]';  
    end
    
    fv = [a/abs(a-1) * exp(-1) / abs(z(i))]';             % initial jump sizes: not normalized density, maximum in tab(y) = z 
    if ((a==2) && (abs(z(i))>2))                          % initial jump sizes: case alpha == 2, stationary points at the bounding of the domain 
        fv = [0.5 * exp(-((abs(z(i)/2))^2)) * abs(z(i))]';
    end
    
    while accept == false
        u1 = unifrnd(0,1); 
          
        p = [fv(1:end).*diff(jt)]; % areas 
        p = p/sum(p);              % normalization step  
        x1 = mnrnd(1, p);          % 1 sample from a multinomial with parameters p: select the interval
        x = unifrnd(jt(x1==1),jt(find(x1==1)+1)); % x is oniformly sampeled on the selected interval
          
        %AR STANDARD
        if  u1<=f_yz(x,z(i),a,b) / fv(find((jt>=x), 1, 'first') - 1 ); 
            y(i) = x; 
            accept = true;
        else %ADAPTING THE DENSITY <=> adapting jump points and values 
            ind = find((jt<=x), 1, 'last' );  
            jt = [jt(1:(ind)); x; jt((ind+1):end)];   
              
            if (tab(x,a,b)<=z(i)) % the new value is on the LEFT of x_max, 
                fv = [fv(1:(ind-1)); f_yz(x,z(i),a,b); fv((ind):end)]; 
            else                  % the new value is on the RIGHT of x_max,
                fv = [fv(1:(ind)); f_yz(x,z(i),a,b); fv((ind+1):end)]; 
            end 
        end
    end 
end 
    
    
    
    
    
    