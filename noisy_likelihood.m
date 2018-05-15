% noisy_likelihood - computes a noisy unbiased estimates of the
% log-likelihood by drawing gl2 iid latent variables from an adaptive
% envelope proposal, built with gl1 grid points 
%
% INPUT:
% a   - alpha
% b   - beta2
% z   - standard stable dataset, parametrization S1 (b2,s2) 
% N   - length(z)
% gl1 - G = number of points used to build the piecewise constant envelope
% gl2 - M = number of points simulated from the piecewise constant envelope
% 
% OUTPUT:
% NL  - value of the noisy unbiased log-lik estimate produced by importance sampling, with the values of alpha and beta in input 
% 
% EXAMPLE:
% NL = noisy_likelihood(1.2, 0.5, CMS_weron(1.2, 1, 0.5, 0, 100), 100, 20, 50) 


function NL = noisy_likelihood(alpha, beta, z, N, gl1, gl2) 

% inline functions 
eab = @(al,bl) (bl .* pi/2 .* al) .*(0<al & al<1) + ( bl .* pi/2 .* al - bl.*pi ) .* (1<al & al<2);  
tab = @(yl,al,bl) sin( pi.* al .*yl + eab(al,bl)) ./ ((cos( pi* yl )).^(1./al)) .* (cos( pi.* (al-1).*yl + eab(al,bl)) ).^((1-al)./al);  
lab = @(al,bl) -eab(al,bl)./(pi.*al);
f_yx = @(yl,xl,al,bl,sl,ml)  1./sl * al./abs(al-1).*  (exp( -(( abs( ((xl-ml)./sl)./tab(yl,al,bl) ) ).^(al./(al-1))) ) )  .* ((abs(((xl-ml)./sl)./tab(yl,al,bl))).^(al./(al-1)))./abs(((xl-ml)./sl));

log_f_yx = @(yl,xl,al,bl,sl,ml)(-log(sl) + log(al./abs(al-1)) ...
                                -(( abs( ((xl-ml)./sl)./tab(yl,al,bl) ) ).^(al./(al-1)))...
                                +1./(al-1).*log(abs((xl-ml)./sl))...
                                -al./(al-1).*log(abs(sin( pi.* al .*yl + eab(al,bl))))...
                                +1./(al-1)*log(abs(cos( pi* yl )))...
                                +log(abs(cos( pi.* (al-1).*yl + eab(al,bl)) ))); 

% storing the space for likelihood terms corresponding to each data point 
lpz = nan(N, 1);

% storing jump points and function values used to build the adaptive piecewise constant envelopes
jt = cell(1, N); % each element is a gl1 x 1 vector 
fv = cell(1, N); % each element is a gl1 x 1 vector 

for i=1:N   % for each stable data-point
% 1) building the adaptive proposal for y_j 
    
    % initial jump points (equivalent to the grid domain)
    if z(i)>=0      
        jt{1,i} = [lab(alpha,beta), 1/2]';  
    else
        jt{1,i} = [-1/2, lab(alpha,beta)]';  
    end
   
    % rest of the grid, built with multinomial sampling 
    fv{1,i} = alpha/abs(alpha-1) * exp(-1) / abs(z(i));       %initial jump sizes: not normalized density, maximum in tab(y) = z 
    if ((alpha==2) && (abs(z(i))>2))                          %initial jump sizes: rare case alpha == 2, stationary points at the boundaries of the domain 
        fv{1,i} = 0.5 * exp(-((abs(z(i)/2))^2)) * abs(z(i));
    end
    
    for j=1:(gl1-2 + 1) 
        p = fv{1,i}(1:end).*diff(jt{1,i}); %areas of piecewise constant traits 
        p = p/sum(p);                      % normalization step  
        x1 = mnrnd(1, p);                  % 1 sample from a multinomial with parameters p: select the interval
        x = unifrnd(jt{1,i}(x1==1),jt{1,i}(find(x1==1)+1)); % x is uniformly sampled from the selected interval 
          
        % insert the new jump point
        ind = find((jt{1,i}<=x), 1, 'last' ); 
        jt{1,i} = [jt{1,i}(1:(ind)); x; jt{1,i}((ind+1):end)];  
              
        % insert the new jump value, either to the left or to the right of the mode 
        if (tab(x,alpha,beta)<=z(i))     % the new value is on the LEFT of x_max, 
            fv{1,i} = [fv{1,i}(1:(ind-1)); f_yx(x,z(i),alpha,beta,1,0); fv{1,i}((ind):end)]; 
        else                              % the new value is on the RIGHT of x_max,
            fv{1,i} = [fv{1,i}(1:(ind)); exp(log_f_yx(x,z(i),alpha,beta,1,0)); fv{1,i}((ind+1):end)]; 
        end 
    end
    
% 2) multinomial sampling of gl2 y_j values from the piecewise constant f_grid, using a robust procedure 
       p = fv{1,i}(1:end).*diff(jt{1,i}); %final areas 
       AP = sum(p);                       % normalizing constant
       p = p/AP;                          % normalization step  
       m = mnrnd(gl2, p);                 % gl2 samples from a multinomial with parameter p 

       A = jt{1,i}(1:end-1); % left extremes  
       B = jt{1,i}(2:end);   % right extremes 
     
       % draw from the piecewise constant envelope, and compute value of the envelope
       y = cell(1,gl1);
       fy = cell(1,gl1); 
       for k = 1: gl1
         y{1,k} = (sort(unifrnd( A(k), B(k), m(k), 1 )))';
         fy {1,k} =  (repmat(fv{1,i}(k), m(k), 1 ))';
       end

       % remove empty cells and concatenating in an array
       Y = cell2mat( y(~cellfun('isempty',y)) ) ; 
       FY = cell2mat(fy(~cellfun('isempty',fy)) ) ; 
       
% 3) compute the likelihood term corresponding to each data point in a robust way (subtracting the maximum)
       log_tilde_C = max(log_f_yx(Y,z(i),alpha,beta,1,0) - log(FY)); 
       lpz(i) = log_tilde_C + log(AP) + log(sum(exp( log_f_yx(Y, z(i), alpha, beta, 1, 0)  - log(FY) - log_tilde_C  ))) - log(gl2);
end  
    
% 4) noisy unbiased log-likelihood 
NL = sum(lpz);

  
  
  
  
  