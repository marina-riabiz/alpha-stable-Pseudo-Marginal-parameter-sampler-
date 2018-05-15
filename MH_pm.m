% MH_pm  - Compute Markov Chains for the parameters 
%  using a pseudo-marginal Gibbs sampler scheme  
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
% gl1      - number of internal gridpoints used to define the levels of each adaptive, piecewise constant, envelope ptoposal distribution for the latent variables y 
% gl2      - number of samples drawn from the final adaptive enveloep, and used for the importance sampler 
% do_plot  - flag, 1 to show the trace plot of the chains 
% 
% OUTPUT:
% out - Ncx3 matrix  - out{:,1} - chain for alpha
%                    - out{:,2} - chain for beta  
%                    - out{:,3} - log-likelihood numeric approximation 
% acc                - Ncx1 vector, 1 if the proposed alpha has been accepted, 0 otherwise  
% cov_post           - sample posterior covariance estimated in the short run, and used to set the paremeters for the long run (alpha and beta not independent) 
% EXAMPLE: 
% [out, acc, cov_post] = MH_pm(sort(CMS_weron(1.2, 1, 0.8, 0, 100)), 100, 1000, 500, 1000, 1.7, 0.4, 0.85, sqrt(1e-3), 50, 100, 1)


function [out, acc, cov_post] = MH_pm(z, N, Ncs, burn,  Nc, alpha_0, beta_0, p, sigma, gl1, gl2, do_plot)

%% 1) initial parameters and functions 

%% 2) SHORT RUN to determine the posterior covariance

% alpha, beta, pseudo-marginal log-lik 
out = nan(Ncs,3); 
out(1,1) = alpha_0;    
out(1,2) = beta_0;   
out(1,3) = noisy_likelihood(out(1,1), out(1,2), z, N, gl1, gl2); % pseudo_marginal_0

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
        % I MH with uniform proposals 
        ap = unifrnd(lo_a,hi_a);    
        bp = unifrnd(lo_b,hi_b);
    end

         
    if  (ap < lo_a || ap > hi_a) || (bp < lo_b || bp > hi_b)               % proposed values out of the bounds <-> reject
    out(k,1)= out(k-1,1);    
    out(k,2)= out(k-1,2);    
    out(k,3)= out(k-1,3);   
    
    else    %acceptance test
    
    % pseudo-marginal log lik in the proposed value 
    zp = noisy_likelihood(ap, bp, z, N, gl1, gl2);  
       
    u = rand(1);
    
    if u<(exp(zp - out(k-1,3)))  % accept 
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

% alpha, beta, pseudo-marginal log-lik 
out = nan(Nc,3); 
out(1,1) = alpha_0;  
out(1,2) = beta_0;   
out(1,3) = noisy_likelihood(out(1,1), out(1,2), z, N, gl1, gl2);

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
        [X] = mvnrnd([out(k-1,1),out(k-1,2)],S); %rmvnrnd([out(k-1,2),out(k-1,1)], S, 1, [-eye(2); eye(2)],  [-lo_b; -lo_a; hi_b; hi_a]);
        ap = X(1); bp = X(2);
    else
        % independent MH with uniform proposals 
        ap = unifrnd(lo_a,hi_a);    
        bp = unifrnd(lo_b,hi_b);
    end

    if  (ap < lo_a || ap > hi_a) || (bp < lo_b || bp > hi_b)               % proposed values out of the bounds <-> reject
    out(k,1)= out(k-1,1);    
    out(k,2)= out(k-1,2);    
    out(k,3)= out(k-1,3);   
    
    else    %acceptance test
    
    % pseudo-marginal log-lik in the proposed value 
    zp = noisy_likelihood(ap, bp, z, N, gl1, gl2);  
       
    u = rand(1);
    
    if u<(exp(zp - out(k-1,3))) % accept
        out(k,1)= ap;    
        out(k,2)= bp;    
        out(k,3)= zp;   
        acc(k) = 1; 
    else
        out(k,1)= out(k-1,1);   % reject
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

