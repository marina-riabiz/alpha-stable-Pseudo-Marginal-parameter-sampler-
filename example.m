% INPUT 
% enter a number in {1,2,3,4,5,6} to choose which MCMC scheme to use
% 1 for conditional_v
% 2 for conditional_y
% 3 for GS_approx_marg
% 4 for MH_approx_marg
% 5 for GS_pm
% 6 for MH_pm 

clear all; close all; clc; 
% dataset
a = 1.2;   % alpha 
s = 1;     % sigma2
b = 0.8;   % beta2
m = 0;     % mu 
N = 1e2;   % sample size 
z = CMS_weron(a, s, b, m, N);  % stable data 

% chains 
Nc = 1e3;                                              % chain lenght
Ncs = 1e3; burn = 5e2;                                 % short run lenght and burn-in when a MH scheme is used 
alpha_0 = 1.7; beta_0 = 0.4;                           % initial values                
p_a = 0.85; p_b = 0.85; p = 0.85;                      % gaussian random walk components probabilities
s_a = sqrt(1e-3); s_b = sqrt(1e-3); sigma= sqrt(1e-3); % gaussian random walk components standard deviations 
MH_steps = 10;                                         % number of MH within Gibbs steps 
do_plot = 1;                                           % to show teh traceplot of the chains 
gl1 = 20;                                              % number of points used to build the adaptive envelope as proposal for the latent variables
gl2 = 50;                                              % number of latent variables sampled for the importance sampler


n = input('Enter a number in {1,2,3,4,5,6}: ');

switch n
    case 1
        disp('conditional_v')
        [out, acc_a, acc_b] = conditional_v(z, N, Nc, alpha_0, beta_0, p_a, p_b, s_a, s_b, MH_steps, do_plot)
    case 2
        disp('conditional_y')
        [out, Nc, acc_a, acc_b] = conditional_y(z, N, Nc, alpha_0, beta_0, p_a, p_b, s_a, s_b,  MH_steps, do_plot)
    case 3
        disp('GS_approx_marg')
        [out, acc_a, acc_b] = GS_approx_marg(z, N, Nc, alpha_0, beta_0,  p_a, p_b, s_a, s_b, do_plot)
    case 4
        disp('MH_approx_marg')
        [out, acc, cov_post] = MH_approx_marg(z, N, Ncs, burn,  Nc, alpha_0, beta_0, p, sigma, do_plot)
    case 5
        disp('GS_pm')
        [out, acc_a, acc_b] = GS_pm(z, N, Nc, alpha_0, beta_0, p_a, p_b, s_a, s_b, gl1, gl2, do_plot)
    case 6
        disp('MH_pm')
        [out, acc, cov_post] = MH_pm(z, N, Ncs, burn,  Nc, alpha_0, beta_0, p, sigma, gl1, gl2, do_plot)
    otherwise
        disp('error: enter a number in {1,2,3,4,5,6}')
end 

