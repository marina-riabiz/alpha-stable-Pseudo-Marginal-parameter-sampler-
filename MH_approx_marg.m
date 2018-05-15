% MH_approx_marg  - Compute Markov Chains for the parameters 
%  using an approximate Metropolis Hastings marginal scheme, based on Nolan's stable density function.  
%  It uses the y-parametrization. Outputs also the acceptance vectors 

% INPUT:
% z        - standard stable dataset; not necessary to pre-order it 
% N        - length(z)
% Ncs      - length of the short run of the chains
% burn     - burn-in done on the short run, before computing the sample posterior covariance matrix 
% Nc       - length of the long run of the chains
% alpha_0  - initial value for the chain on alpha 
% beta_0   - initial value for the chain on alpha 
% p        - probability of the truncated gaussian random walk in the proposal (independent uniform sampling otherwise )
% s        - standard deviation of the truncated gaussian random walk in the proposal 
% do_plot  - flag, 1 to show the trace plot of the chains 
% 
% OUTPUT:
% out - Ncx3 matrix  - out{:,1} - chain for alpha
%                    - out{:,2} - chain for beta  
%                    - out{:,3} - log-likelihood numeric approximation 
% acc                - Ncx1 vector, 1 if the proposed alpha has been accepted, 0 otherwise  
% cov_post           - sample posterior covariance estimated in the short run, and used to set the paremeters for the long run (alpha and beta not independent) 
% EXAMPLE: 
% [out, acc, cov_post] = MH_approx_marg(sort(CMS_weron(1.2, 1, 0.8, 0, 100)), 100, 1000, 500, 1000, 1.7, 0.4, 0.85, sqrt(1e-3), 1)

                                                      
function [out, acc, cov_post] = MH_approx_marg(z, N, Ncs, burn,  Nc, alpha_0, beta_0, p, sigma, do_plot)

%% 1) initial parameters and functions 

% to call stablepdf we need the S^1(\sigma, \beta) parametrization
b2_to_b = @(al, bl) ((tan(bl.*pi.*al./2)./tan(pi.*al./2)) .* (0<al & al<1) + (tan(bl.*pi.*(al-2)./2)./tan(pi.*al./2)) .* (1<al & al<2));  
s2_to_s = @(al, sl, bl) (sl ./((1+  ((b2_to_b(al,bl)).^2) .* ((tan(pi.*al./2)).^2) ).^(1./(2.*al)))); 
s = 1; m = 0; % (\sigma_2, \mu_2) parameters of the stable standard data

%% 2) SHORT RUN to determine the posterior covariance

% alpha, beta, log-lik approximation 
out = nan(Ncs,3); 
out(1,1) = alpha_0;   
out(1,2) = beta_0;   

% alpha_proposal params
if  out(1,1)>1
    lo_a= 1.1;   
    hi_a = 2; 
else
    lo_a= 0.1;   
    hi_a = 0.9;     
end
% beta_proposal param
if out(1,2)>0
    lo_b = 0; 
    hi_b = 1; 
else
    lo_b = -1; 
    hi_b = 0; 
end
% covariance for the short run: alpha, beta independent
s_a = sigma^2;  
s_b = sigma^2;  
rho = 0;
cv = rho*(sqrt(s_a)*sqrt(s_b));
S = [ s_a cv; cv s_b ]; 

% store space for the intermediate approximated log-likelihood in the proposed and old param
zpi= nan(N,1);

zoi= nan(N,1);
for i = 1: N 
    zoi(i) = stablepdf(z(i), [out(1,1), b2_to_b(out(1,1),out(1,2)), s2_to_s(out(1,1), s, out(1,2)) , m], 1);
end
out(1,3) = sum(log(zoi)); 

% storing acc rate  
acc = zeros(Ncs,1); 

% MH scheme + plot
if do_plot ==1
figure()
end

for k = 2:Ncs
    % proposal (alpha, beta): is a mixture: truncated 2-dim gaussian(random walk component)  + truncated 2-dim uniform(independent component): 
    mixt = binornd(1,p);

    if mixt==1
        % untruncated multivariate gaussian  
        [X] = mvnrnd([out(k-1,1),out(k-1,2)],S); 
        ap = X(1); bp = X(2);
    else
        % independent MH with uniform proposals 
        ap = unifrnd(lo_a,hi_a);    
        bp = unifrnd(lo_b,hi_b);
    end

    if  (ap < lo_a || ap > hi_a) || (bp < lo_b || bp > hi_b)  % proposed values out of the bounds <-> reject
    out(k,1)= out(k-1,1);    
    out(k,2)= out(k-1,2);    
    out(k,3)= out(k-1,3);   
        
    else  %acceptance test

    % approximated log-lik in the proposed values 
    for i = 1: N
      zpi(i) = stablepdf(z(i), [ap, b2_to_b(ap,bp), s2_to_s(ap, s, bp) , m], 1);
    end
    zp = sum(log(zpi));       
        
    u = rand(1);
    
    if u<(exp(zp -  out(k-1,3))) % accept 
        out(k,1)= ap;    
        out(k,2)= bp;    
        out(k,3)= zp;    
        acc(k) = 1; 
    else
        out(k,1)= out(k-1,1);    % reject
        out(k,2)= out(k-1,2);    
        out(k,3)= out(k-1,3);    
    end

    end

    % PLOT of the short run    
    if do_plot==1
    %alpha
    subplot(3,1,1)
    hold on;
    plot(k, out(k,1), 'b.-'); 
    xlabel('iterations')
    ylabel('$\alpha$', 'Interpreter', 'latex')
    set(gca, 'FontSize', 10)        
    %beta
    subplot(3,1,2)
    hold on;
    plot(k, out(k,2), 'b.-');
    xlabel('iterations')
    ylabel('$\beta$', 'Interpreter', 'latex')        
    % approximated log-likelihood
    subplot(3,1,3)
    hold on;
    plot(k, out(k,3), 'b.-');
    xlabel('iterations')
    ylabel('$\hat{z}$', 'Interpreter', 'latex')        
    pause(1e-15);
    end
end

% posterior covariance from the short run
cov_post = cov(out((burn+1):Ncs, 1:2)); 


%% 3) LONG RUN: 

% alpha, beta, log-lik approximation 
out = nan(Nc,3); 
out(1,1) = alpha_0; 
out(1,2) = beta_0;    

% alpha_proposal param
if  out(1,1)>1
    lo_a= 1.1;   
    hi_a = 2; 
else
    lo_a= 0.1;   
    hi_a = 0.9;     
end
% beta_proposal param
if out(1,2)>0
    lo_b = 0; 
    hi_b = 1; 
else
    lo_b = -1; 
    hi_b = 0; 
end

S = cov_post; 

% store space for the intermediate approximated log-likelihood in the proposed and old param
zpi= nan(N,1);

zoi= nan(N,1);
for i = 1: N 
    zoi(i) = stablepdf(z(i), [out(1,1), b2_to_b(out(1,1),out(1,2)), s2_to_s(out(1,1), s, out(1,2)) , m], 1);
end
out(1,3) = sum(log(zoi)); 

% storing acc rate  
acc = zeros(Nc,1); 


% MH scheme + plot
if do_plot ==1
figure()
end

for k = 2:Nc
    
    % proposal (alpha, beta): is a mixture: truncated 2-dim gaussian(random walk component)  + truncated 2-dim uniform(independent component): 
    mixt = binornd(1,p);

    if mixt==1
        % untruncated multivariate gaussian  
        [X] = mvnrnd([out(k-1,1),out(k-1,2)],S); 
        ap = X(1); bp = X(2);
    else
        % independent MH with uniform proposals 
        ap = unifrnd(lo_a,hi_a);    
        bp = unifrnd(lo_b,hi_b);
    end

    if  (ap < lo_a || ap > hi_a) || (bp < lo_b || bp > hi_b)   % proposed values out of the bounds <-> reject
    out(k,1)= out(k-1,1);    
    out(k,2)= out(k-1,2);    
    out(k,3)= out(k-1,3);   
        
    else  %acceptance test

    % approximated log-lik in the proposed values 
    for i = 1: N
      zpi(i) = stablepdf(z(i), [ap, b2_to_b(ap,bp), s2_to_s(ap, s, bp) , m], 1);
    end
    zp = sum(log(zpi));       

    u = rand(1);
    
    if u<(exp(zp -  out(k-1,3))) % accept
        out(k,1)= ap;    
        out(k,2)= bp;    
        out(k,3)= zp;    
        acc(k) = 1; 
    else
        out(k,1)= out(k-1,1);    % reject
        out(k,2)= out(k-1,2);    
        out(k,3)= out(k-1,3);    
    end

    end
    
    % PLOT of the long chain     
    if do_plot==1
    %alpha
    subplot(3,1,1)
    hold on;
    plot(k, out(k,1), 'b.-'); 
    xlabel('iterations')
    ylabel('$\alpha$', 'Interpreter', 'latex')
    set(gca, 'FontSize', 10)        
    %beta
    subplot(3,1,2)
    hold on;
    plot(k, out(k,2), 'b.-');
    xlabel('iterations')
    ylabel('$\beta$', 'Interpreter', 'latex')        
    % approximated log-likelihood
    subplot(3,1,3)
    hold on;
    plot(k, out(k,3), 'b.-');
    xlabel('iterations')
    ylabel('$\hat{z}$', 'Interpreter', 'latex')        
    pause(1e-15);
    end
    
end





