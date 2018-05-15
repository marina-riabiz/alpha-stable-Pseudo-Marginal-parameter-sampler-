% safe_newton - root finding via newton algorithm, 
%               where a bisection step is taken each time 
%               the corresponding interval is not bracketing 
%
% INPUT:
% fun  - (scalar and one variable) function handle 
% der  - derivative handle 
% a, b - initial interval; must be bracketing the solution  
% tx   - tol on the lenght of the bracketing interval (stopping criterium)
% tf   - tol on the values of f (stopping criterium)
% nMax - maximum number of iterations (stopping criterium)
%
% OUTPUT:
% x      - approximation of the zero of f  
% xEvals - vector of successive zero approximations 
% test   - boolean, 0 if the initial interval is bracketing the zero, 1 otherwise 
% nEvals - number of effective iterations 
%
% NOTES:
% The iterations terminate as soon as x is within a tolerenace tx of a true zero or
% if |f(x)|<= tf or after nMax itartions 
%
% EXAMPLE:
% fun = @(x)(x-1); der = @(x)(1); a= 0.5; b = 1.5; tx= 1e-4; tf = 1e-4; nMax = 1e3; 
% x = safe_newton(fun,der,a,b,tx,tf,nMax)

function [x,xEvals,test,nEvals] = safe_newton(fun,der,a,b,tx,tf,nMax)
% check if the given interval (a,b) is bracketing. Test for a monotonic function 
test=0;  
fa = fun(a); 
fb = fun(b);  
if fa*fb>0 
    test=1;
    x = [];
    error('Initial interval not bracketing.');  
end

%store evaluated points
xEvals = nan(nMax,1); 

% initial guess = one of the extremes of the bracketing interval  
x = a;
fx=fun(x); fpx=der(x);  
nEvals = 1;
xEvals(1)=x; 
  
while (abs(a-b)>tx) && (abs(fx) > tf) && ((nEvals<nMax-1) || (nEvals==1)) % tolerance on x, on f and on the number of iterations 
    
    % Newton's guess
    x1=x-fx/fpx;
    
    if (((fa*fun(x1)<0) || (fb*fun(x1)<0)) && (a<x1) && (x1<b)) 
        %Take Newton Step
        x = x1;
        xEvals(nEvals+1)=x; 
    else
        %Take a Bisection Step
        x = (a+b)/2;
        xEvals(nEvals+1)=x; 
    end
    
    fx = fun(x);
    fpx = der(x);
    nEvals = nEvals+1;

    % steps done in every case 
    if fa*fx<=0
        % there is a root in [a,x]. Bring in right endpoint.
        b = x;
        fb = fx;
    else
        % there is a root in [x,b]. Bring in left endpoint.
        a = x;
        fa = fx;
    end
end



