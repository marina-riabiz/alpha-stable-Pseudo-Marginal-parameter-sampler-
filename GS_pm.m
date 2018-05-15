% GS_pm  - Compute Markov Chains for the parameters 
%  using a pseudo-marginal Gibbs sampler scheme  
%  It uses the y-parametrization. Outputs also the acceptance vectors 

% INPUT:
% z        - standard stable dataset; not necessary to pre-order it 
% N        - length(z)
% Nc       - length of the chains
% alpha_0  - initial value for the chain on alpha 
% beta_0   - initial value for the chain on alpha 
% p_a      - probability of the truncated gaussian random walk in the proposal for alpha full-conditional (independent uniform sampling otherwise )
% p_b      - probability of the truncated gaussian random walk in the proposal for beta full-condituional (independent uniform sampling otherwise )
% s_a      - standard deviation of the truncated gaussian random walk in the proposal for the alpha full-conditional 
% s_b      - standard deviation of the truncated gaussian random walk in the proposal for the beta full-conditional 
% gl1      - number of internal gridpoints used to define the levels of each adaptive, piecewise constant, envelope ptoposal distribution for the latent variables y 
% gl2      - number of samples drawn from the final adaptive enveloep, and used for the importance sampler 
% do_plot  - flag, 1 to show the trace plot of the chains 
% 
% OUTPUT:
% out - Ncx4 matrix  - out{:,1} - chain for alpha
%                    - out{:,2} - chain for beta  
%                    - out{:,3} - pseudo-marginal log-likelihood after the step on beta, in the GS  
%                    - out{:,4} - pseudo-marginal log-likelihood after the step on alpha, in the GS  
% acc_a              - Ncx1 vector, 1 if the proposed alpha has been accepted, 0 otherwise  
% acc_b              - Ncx1 vector, 1 if the proposed beta has been accepted, 0 otherwise
% 
% EXAMPLE: 
% [out, acc_a, acc_b] = GS_pm(sort(CMS_weron(1.2, 1, 0.8, 0, 100)), 100, 1000, 1.7, 0.4, 0.85, 0.85, sqrt(1e-3), sqrt(1e-3), 50, 100, 1)


function [out, acc_a, acc_b] = GS_pm(z, N, Nc, alpha_0, beta_0, p_a, p_b, s_a, s_b, gl1, gl2, do_plot)

%% 1) initial parameters and functions 

% alpha, beta, log-lik approximation after beta step, log-lik approximation after alpha step 
out = nan(Nc,4);  
out(1,1) = alpha_0;    
out(1,2) = beta_0;   
out(1,3) = noisy_likelihood(out(1,1), out(1,2), z, N, gl1, gl2); % pseudo-marginal_0 log-lik after beta step

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
    lo_b= 0; 
    hi_b = 1; 
else
    lo_b= -1; 
    hi_b = 0; 
end


% storing acc rates  
acc_a = zeros(Nc,1); 
acc_b = zeros(Nc,1); 

%% 2) Gibbs pseudo-marginal scheme iterations + trace-plots 

if do_plot ==1
figure()
end

for t = 2:Nc
    
    % 2.1) ALPHA
    % alpha-proposal:  is a mixture: truncated gaussian(random walk component) + uniform(independent component):
    mixt = binornd(1, p_a);
    if mixt==1
        % ap sampled from a truncated gaussian, centered on previous alpha-value
        plo = normcdf((lo_a-out(t-1,1))/s_a);  
        phi = normcdf((hi_a-out(t-1,1))/s_a); 
        r = unifrnd(plo, phi);
        ap = norminv(r);
        ap = out(t-1,1) + ap*s_a;
        % proposal distribution: the acceptance ratio requires the normalizing constant 
        alpha_prop = @(m)(normcdf((hi_a-m)/s_a)-normcdf((lo_a-m)/s_a));
    else
        ap = unifrnd(lo_a,hi_a);
        alpha_prop = @(m)1;
    end
    % 2.2) PSEUDO-MARGINAL LOG-LIK 
    zp = noisy_likelihood(ap, out(t-1,2), z, N, gl1, gl2); 
    % acceptance test:
    u = rand(1);
    if u<( exp(zp - out(t-1,3)) * alpha_prop(ap) / alpha_prop(out(t-1,1)) )
        out(t,1)= ap;          % accept
        out(t,4)= zp;   
        acc_a(t) = 1; 
    else
        out(t,1)= out(t-1,1);  % reject
        out(t,4)= out(t-1,3);    
   end
    
    
    % 2.3) BETA
    % beta-proposal: is a mixture: truncated gaussian(random walk component) + uniform(independent component):
    mixt = binornd(1,p_b);
    if mixt==1
        % bp sampled from a truncated gaussian, centered on previous beta-value
        plo = normcdf((lo_b-out(t-1,2))/s_b);  
        phi = normcdf((hi_b-out(t-1,2))/s_b); 
        r = unifrnd(plo, phi);
        bp = norminv(r);
        bp = out(t-1,2) + bp*s_b;
        % proportional proposal RW
        beta_prop = @(m)(normcdf((hi_b-m)/s_b)-normcdf((lo_b-m)/s_b));
    else
        bp = unifrnd(lo_b,hi_b);
        beta_prop = @(m)1;
    end
    % 2.4) PM-LOG-PROPOSAL: 
    zp = noisy_likelihood(out(t,1), bp, z, N, gl1, gl2); 
    % acceptance test 
    u = rand(1);
    if u<( exp(zp - out(t,4)) * beta_prop(bp) / beta_prop(out(t-1,2)) )
        out(t,2)= bp;    
        out(t,3)= zp; 
        acc_b(t) = 1; 
    else
        out(t,2)= out(t-1,2);    
        out(t,3)= out(t,4);    
    end
 
    
%% 3) PLOT    
    if do_plot ==1 
    %alpha
    subplot(3,1,1)
    hold on;
    plot(t, out(t,1), 'b.-'); 
    xlabel('iterations')
    ylabel('$\alpha$', 'Interpreter', 'latex')
    set(gca, 'FontSize', 10)    
    %beta
    subplot(3,1,2)
    hold on;
    plot(t, out(t,2), 'b.-');
    xlabel('iterations')
    ylabel('$\beta$', 'Interpreter', 'latex')    
    %approximate log-lik
    subplot(3,1,3)
    hold on;
    plot(t, out(t,3), 'b.-');
    xlabel('iterations')
    ylabel('$\hat{z}$', 'Interpreter', 'latex')    
    pause(1e-15);
    end    
    
end

hold off;



