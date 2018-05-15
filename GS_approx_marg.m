% GS_approx_marg  - Compute Markov Chains for the parameters 
%  using an approximate Gibbs sampler marginal scheme, based on Nolan's stable density function.  
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
% do_plot  - flag, 1 to show the trace plot of the chains 
% 
% OUTPUT:
% out - Ncx4 matrix  - out{:,1} - chain for alpha
%                    - out{:,2} - chain for beta  
%                    - out{:,3} - log-likelihood numeric approximation after the step on beta, in the GS  
%                    - out{:,4} - log-likelihood numeric approximation after the step on alpha, in the GS  
% acc_a              - Ncx1 vector, 1 if the proposed alpha has been accepted, 0 otherwise  
% acc_b              - Ncx1 vector, 1 if the proposed beta has been accepted, 0 otherwise
% 
% EXAMPLE: 
% [out, acc_a, acc_b] = GS_approx_marg(sort(CMS_weron(1.2, 1, 0.8, 0, 100)), 100, 1000, 1.7, 0.4, 0.85, 0.85, sqrt(1e-3), sqrt(1e-3), 1)


function [out, acc_a, acc_b] = GS_approx_marg(z, N, Nc, alpha_0, beta_0,  p_a, p_b, s_a, s_b, do_plot)

%% 1) initial parameters and functions 

% to call stablepdf we need the S^1(\sigma, \beta) parametrization
b2_to_b = @(al, bl) ((tan(bl.*pi.*al./2)./tan(pi.*al./2)) .* (0<al & al<1) + (tan(bl.*pi.*(al-2)./2)./tan(pi.*al./2)) .* (1<al & al<2));  
s2_to_s = @(al, sl, bl) (sl ./((1+  ((b2_to_b(al,bl)).^2) .* ((tan(pi.*al./2)).^2) ).^(1./(2.*al)))); 
s = 1; m = 0; % (\sigma_2, \mu_2) parameters of the stable standard data

% alpha, beta, log-lik approximation after beta step, log-lik approximation after alpha step 
out = nan(Nc,4); 
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
    lo_b= 0; 
    hi_b = 1; 
else
    lo_b= -1; 
    hi_b = 0; 
end

% store space for the intermediate approximated log-likelihood in the proposed and old param
zpi= nan(N,1);

zoi= nan(N,1);
for i = 1: N 
    zoi(i) = stablepdf(z(i), [out(1,1), b2_to_b(out(1,1),out(1,2)), s2_to_s(out(1,1), s, out(1,2)) , m], 1);
end
out(1,3) = sum(log(zoi)); 

% storing acc rates  
acc_a = zeros(Nc,1); 
acc_b = zeros(Nc,1);


%% 2) Gibbs approximate marginal scheme iterations + trace-plots 

if do_plot ==1
figure()
end

for t = 2:Nc
    
    % 2.1) ALPHA
    % alpha-proposal:  is a mixture: truncated gaussian(random walk component) + uniform(independent component):
    mixt = binornd(1,p_a);
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
        ap = unifrnd(lo_a, hi_a);
        alpha_prop = @(m)1;
    end
    % 2.2) APPROXIMATE LOG-LIK:  
    for i = 1: N
      zpi(i) = stablepdf(z(i), [ap, b2_to_b(ap,out(t-1,2)), s2_to_s(ap, s, out(t-1,2)) , m], 1);
    end
    zp = sum(log(zpi));      
    % acceptance test 
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
    % 2.4) APPROXIMATE LOG-LIK:  
    for i = 1: N
      zpi(i) = stablepdf(z(i), [out(t,1), b2_to_b(out(t,1),bp), s2_to_s(out(t,1), s, bp) , m], 1);
    end
    zp = sum(log(zpi));      
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

