% conditional_v - Compute Markov Chains for the parameters in Buckle's conditional scheme
% in the v-parametrization. Outputs also the acceptance vectors 

% INPUT:
% z        - standard stable dataset; for computational speed it is better to pre-order it in ascending order (but not necessary)
% N        - length(z)
% Nc       - length of the chains
% alpha_0  - initial value for the chain on alpha 
% beta_0   - initial value for the chain on alpha 
% p_a      - probability of the truncated gaussian random walk in the proposal for alpha full-conditional (independent uniform sampling otherwise )
% p_b      - probability of the truncated gaussian random walk in the proposal for beta full-condituional (independent uniform sampling otherwise )
% s_a      - standard deviation of the truncated gaussian random walk in the proposal for the alpha full-conditional 
% s_b      - standard deviation of the truncated gaussian random walk in the proposal for the beta full-conditional 
% MH_steps - number of internal MH steps
% do_plot  - flag, 1 to show the trace plot of the chains 
% 
% OUTPUT:
% out - Ncx3 cell array - out{:,1} - latent variables y (before the transformation)
%                       - out{:,2} - chain for alpha
%                       - out{:,3} - chain for beta              
% acc_a                 - Ncx1 vector, 1 if the proposed alpha has been accepted, 0 otherwise  
% acc_b                 - Ncx1 vector, 1 if the proposed beta has been accepted, 0 otherwise
% 
% EXAMPLE: 
% [out, acc_a, acc_b] = conditional_v(sort(CMS_weron(1.2, 1, 0.8, 0, 100)), 100, 1000, 1.7, 0.4, 0.85, 0.85, sqrt(1e-3), sqrt(1e-3), 10, 1)


function [out, acc_a, acc_b] = conditional_v(z, N, Nc, alpha_0, beta_0, p_a, p_b, s_a, s_b, MH_steps, do_plot)

%% 1) initial parameters and functions 

% Y, alpha, beta
out = cell(Nc,3);  

% initial values of the chain for the parameters 
out{1,2} = alpha_0;  
out{1,3} = beta_0;      

% alpha_proposal boundary parameters
if  out{1,2}>1
    lo_a= 1.1;   
    hi_a = 2; 
else
    lo_a= 0.1;   
    hi_a = 0.9;     
end

% beta_proposal boundary parameters 
if out{1,3}>0
    lo_b= 0; 
    hi_b = 1; 
else
    lo_b= -1; 
    hi_b = 0; 
end

% storing acceptance rates  
acc_a = zeros(Nc,1); 
acc_b = zeros(Nc,1); 

% general functions
eab = @(al,bl) (bl .* pi/2 .* al) .*(0<al && al<1) + ( bl .* pi/2 .* al - bl.*pi ) .* (1<al && al<2);  
tab = @(yl,al,bl) sin( pi.* al .*yl + eab(al,bl)) ./ ((cos( pi* yl )).^(1./al)) .* (cos( pi.* (al-1).*yl + eab(al,bl)) ).^((1-al)./al);  

% log_jacobian term (log(d_tab/d_y))
log_tab_der = @(yl,al,bl)(log(al * pi) + ...
                         ((al-1)/al) * log(cos(pi*yl)) + ...
                              (1/al) * log(cos( (al-1)*pi*yl + eab(al,bl))) + ...
                                       log(1 + 1/(al^2) .* (tan(pi*yl) - (al-1).* tan((al-1)*pi*yl + eab(al,bl))).^2) );

% alpha_target_log_density in v
log_f_a = @(yl,vl,zl,al,bl) (N*log(al/abs(al-1)) ...
                          - sum( (abs(zl./vl)).^(al/(al-1)) )...
                          + al/(al-1)* sum(  log(abs(zl./vl ))  ) ...
                          - sum(log_tab_der(yl,al,bl)) ); 
% beta_target_log_density in v
log_f_b = @(yl, zl,al,bl) ( - sum(log_tab_der(yl,al,bl)) ); 
  

%% 2) Gibbs conditional scheme iterations + trace-plots 

if do_plot ==1
figure()
end

for t = 2:Nc
       
    % 2.1) Y and ordered V: V corresponds to the sampled Y, alpha, beta from previous step, according to teh function tab            
    out{t,1} = sampling_y(out{t-1,2}, out{t-1,3}, z, N);  
    V = tab(out{t,1}, out{t-1,2}, out{t-1,3}); % V are not in ascending order 
    [VI,I] = sort(V);                          % VI are in ascending order 
    
    % 2.2) ALPHA
    % alpha-proposal: is a mixture: truncated gaussian(random walk component) + uniform(independent component):
    old_a = out{t-1,2};   
    for imh = 1:MH_steps % run internal MH for a number of MH_steps steps 
    mixt = binornd(1,p_a);
    if mixt==1
        % ap sampled from a truncated gaussian, centered on previous alpha-value
        plo = normcdf((lo_a- out{t-1,2})/s_a);  
        phi = normcdf((hi_a- out{t-1,2})/s_a); 
        r = unifrnd(plo, phi);
        ap = norminv(r);
        ap =  out{t-1,2} + ap*s_a;
        % proposal distribution: the acceptance ratio requires the normalizing constant 
        alpha_prop = @(m)(normcdf((hi_a-m)/s_a)-normcdf((lo_a-m)/s_a));
    else
        % ap sampled from a uniform distribution on the interval of interest
        ap = unifrnd(lo_a,hi_a);
        % proposal distribution: simplifies in the acceptance ratio  
        alpha_prop = @(m)1; 
    end
    % new set of y, corresponding to the fixed VI, but new alpha (and old beta); the Y in output have order corresponding to the dataset z 
    Y = inversion_step_ordered(ap, out{t-1,3}, VI, I, N); 
    % acceptance test 
    u = rand(1);
    if u < (exp (log_f_a( Y, V, z, ap, out{t-1,3}) - log_f_a([out{t,1}], V,  z, out{t-1,2}, out{t-1,3}) ) * alpha_prop(ap) / alpha_prop(out{t-1,2}) )        
        old_a = ap; 
    end
    end
    
    out{t,2} = old_a;  % accept the final outcome of MH_steps steps 
    % and eventually increment the counter 
    if out{t-1,2} ~=old_a
         acc_a(t) = 1;
    end 
    
    % 2.3) BETA
    % beta-proposal: is a mixture: truncated gaussian(random walk component) + uniform(independent component):
    old_b = out{t-1,3};   
    for imh = 1:MH_steps  % run internal MH for a number of MH_steps steps 
    mixt = binornd(1,p_b);
    if mixt==1
        % bp sampled from a truncated gaussian, centered on previous beta-value
        plo = normcdf((lo_b-out{t-1,3})/s_b);  
        phi = normcdf((hi_b-out{t-1,3})/s_b); 
        r = unifrnd(plo, phi);
        bp = norminv(r);
        bp = out{t-1,3} + bp*s_b;
        % proposal distribution: the acceptance ratio requires the normalizing constant 
        beta_prop = @(m)(normcdf((hi_b-m)/s_b)-normcdf((lo_b-m)/s_b));
    else
        % bp sampled from a uniform distribution on the interval of interest
        bp = unifrnd(lo_b,hi_b);
        % proposal distribution: simplifies in the acceptance ratio  
        beta_prop = @(m)1;
    end
    % new set of y, corresponding to the fixed VI but new alpha and new beta; the Y in output have order corresponding to the dataset z 
    Y = inversion_step_ordered(out{t,2}, bp, VI, I, N); 
    % acceptance test
    u = rand(1);
    if  u < (exp (log_f_b(Y, z, out{t,2}, bp) - log_f_b([out{t,1}], z, out{t,2}, out{t-1,3}) ) * beta_prop(bp) / beta_prop(out{t-1,3}) )        
        old_b = bp;                
    end
    end
    
    out{t,3} = old_b;  % accept the final outcome of MH_steps steps  
    % and eventually increment the counter 
    if out{t-1,3} ~=old_b
         acc_b(t) = 1;
    end     
    
%% 3) PLOT
    if do_plot ==1
    %alpha
    subplot(2,1,1)
    hold on;
    plot(t, out{t,2}, 'b.-'); 
    xlabel('iterations')
    ylabel('$\alpha$', 'Interpreter', 'latex')
    set(gca, 'FontSize', 10)
    %beta
    subplot(2,1,2)
    hold on;
    plot(t, out{t,3}, 'b.-');
    xlabel('iterations')
    ylabel('$\beta$', 'Interpreter', 'latex')
    pause(1e-15);
    end

end


