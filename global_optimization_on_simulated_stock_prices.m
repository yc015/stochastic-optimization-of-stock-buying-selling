% Simulate the mean-reverting stock prices using 
% the Ornstei-Uhlenbeck process

% Get 100 random trends of stock prices
NSample = 100;
% approximate the stock prices at 100 equal spaced time points
N_T = 100;
% Start at the time 0
t0=0;
% End at the time 100
T = 100;

% Generate 100 random brownian motion using the above setup
[TProcess, BProcess] = BrownMotion(NSample, t0, 0, T, N_T);

% Starting stock price is exp(2).
y0 = [2];
% Compute the step size
h = (T - t0) / (N_T - 1);

stock_prices = [];

% Get 100 random trends of stock prices
% The diff eq for generating trends of stock prices using OU process is
% dX(t) = \lambda (\mu - X(t))dt + \sigma dW(t)
% The stock price is S(t) = exp(X(t))
% \mu is the long-term average
% \lambda is the speed of mean reversion
% \sigma is the volatility of the process
for k=1:NSample
    w_diff = diff(BProcess(:, k));
%     w_diff = normrnd(0, sqrt(h), [N_T-1, 1]);
    x_prime = @(t, y) 0.8 * (2 - y(1)) + 0.5 * (w_diff(int32(t / h) + 1));
    f = {x_prime};
    
    % Approximate 
    [ymat, t] = EulerSystem(f, t0, y0, T, N_T);
    stock_prices = cat(2, stock_prices, exp(ymat(:, 1)));
end

% Plot the simulated stock prices
for k=1:NSample
    plot(t, stock_prices(:, k))
    hold on
end
p = [];
p(1) = plot(t, ones(length(t), 1) * exp(2),'g','LineWidth',3, 'DisplayName', "Long-term Mean");
p(2) = plot(t, ones(length(t), 1) * mean(mean(stock_prices)),'r','LineWidth',3, 'DisplayName', "Stock Price Mean");
title("dX(t) / dt = 0.8(2 - X(t)) + dW(t) / dt",'FontSize', 16)
xlabel("Time", 'FontSize', 16)
ylabel("Price", 'FontSize', 16)
legend(p, 'FontSize', 10)
hold off

% Starting buying threshold
b0 = 8;
% Starting selling threshold
s0 = 15;
% Run for 300 iterations
num_iter = 300;
s = [];
b = [];
reward = [];
b(1) = b0;
s(1) = s0;
% Globally optimize on 100 randomly generated stock prices
% Optimization mode: Unit Gradient (mode == 1)
mode = 1;
strategies = ["Adaptive Gradient", "Unit Gradient", "Diagonal", "Uni-directional"];
for iter=1:num_iter    
    c = (iter) ^ (-1/3);
    reward(iter) = TransactionReward(stock_prices, b(iter), s(iter), T);

    incre_b = b(iter) + c;
    reward_incre_b = TransactionReward(stock_prices, incre_b, s(iter), T);

    decre_b = b(iter) - c;
    reward_decre_b = TransactionReward(stock_prices, decre_b, s(iter), T);

    incre_s = s(iter) + c;
    reward_incre_s = TransactionReward(stock_prices, b(iter), incre_s, T);

    decre_s = s(iter) - c;
    reward_decre_s = TransactionReward(stock_prices, b(iter), decre_s, T);

    gradient_b = (reward_incre_b - reward_decre_b) / (2 * c);
    gradient_s = (reward_incre_s - reward_decre_s) / (2 * c);

    mag = sqrt(gradient_b ^ 2 + gradient_s ^ 2);

    [next_b, next_s] = update_bs(b(iter), s(iter), ...
        gradient_b, gradient_s, ...
        iter, c, mode);

    b(iter + 1) = next_b;
    s(iter + 1) = next_s;

    change = sqrt((b(iter + 1) - b(iter))^2 + (s(iter + 1)- s(iter))^2);
    if iter > 1 && change < 0.001
        break
    end
end
reward(iter + 1) = TransactionReward(stock_prices, b(iter + 1), s(iter + 1), T);

figure()
plot(linspace(0, length(reward) - 1, length(reward)), s)
hold on
plot(linspace(0, length(reward) - 1, length(reward)), b)
plot(linspace(0, length(reward) - 1, length(reward)), reward)
legend("Selling Threshold", "Buying Threshold", "Final Reward" + " " + string(reward(end)), "FontSize", 12)
title("Global Optim with " + strategies(mode + 1), "FontSize", 15)
ylabel("Price", "FontSize", 15)
xlabel("Iterations", "FontSize", 15)
hold off

s_all = linspace(1, 26, 50);
b_all = linspace(1, 26, 50);
reward_all = [];
for i=1:length(s_all)
    for j=1:length(b_all)
        reward_all(j, i) = TransactionReward(stock_prices, b_all(i), s_all(j), T);
    end
end

figure()
h = surf(b_all, s_all, reward_all)
% alpha 0.85
set(h,'LineStyle','none')
xlabel("Buying Threshold")
ylabel("Selling Threshold")
zlabel("Reward")
hold on
p=[];
p(1) = plot3(b(1), s(1), reward(1), 'ogreen','linewidth', 3, 'DisplayName', "Start " + string(reward(1)));
plot3(b, s, reward, 'red','linewidth',2)
p(2) = plot3(b(end), s(end), reward(end), 'oblue','linewidth', 6, 'DisplayName', "End " + string(reward(end)));
text(b(1), s(1), reward(1) + 1, append('Start ', string(reward(1))))
text(b(end), s(end), reward(end) + 1, append('End ', string(reward(end))))
% legend(p, "FontSize", 12)
title("Path of Global Optimization", "FontSize", 15)
hold off

% Function for approximating the solution to a differential equations
% using the Euler Method
function [ymat, t]=EulerSystem(f, t0, y0, b, N)
    ymat = zeros(N, length(y0));
    h = (b - t0) / (N - 1);
    for i=1:length(y0)
        ymat(1, i) = y0(i);
    end
    t(1) = t0;
    for j=2:N
        for i=1:length(y0)
            ymat(j, i) = ymat(j - 1,i) + h * f{i}(t(j-1), ymat(j - 1, :));
            t(j) = t(j - 1) + h;
        end
    end
end

% Function for generating the random Brownian motion
function [TProcess, BProcess]=BrownMotion(sample, t0, y0, b, N)
    BProcess = [];
    TProcess = [];
    NSample = sample;
    N_T = N;
    T = b;
    
    for k = 1:NSample
        B = y0;
        t = t0;
        BProcess(1,k)=B; TProcess(1,k)=t;
        for i=2:N_T
            dt =  T/N_T;
            t = t + dt;
            B = B + sqrt(dt)*randn(1); %increments in B are normally distibuted with variance dt
            BProcess(i,k) = B;
            TProcess(i,k) = t;
        end
    end
end

% Global Optimization
% Optimize on one transaction of multiple different stock prices
function reward=TransactionReward(stock_prices, b, s, T)
    if b > s
        reward = 0;
        return
    end
    phi = [];
    for k=1:length(stock_prices(1, :))
        [tau_b,ColNrs] = find(stock_prices(:, k) < b);
        [tau_s,ColNrs] = find(stock_prices(:, k) > s);
        if length(tau_b) < 1
            continue
        end
        [tau_s,ColNrs] = min(tau_s(find(tau_s > tau_b(1))));
        if length(tau_s) < 1
            continue
        end
        tau_b = tau_b(1);
        phi(k) = exp(-0.01 * tau_s / length(stock_prices) * T) * s * (1 - 0.01) - exp(-0.01 * tau_b / length(stock_prices) * T) * b * (1 + 0.01);
    end
    
    if length(phi) < 1
        reward = 0;
    else
        reward = sum(phi) / length(stock_prices(1, :));
    end       
end

% Function for updating the buying and selling threshold
function [next_b, next_s] = update_bs(b, s, gradient_b, gradient_s, ...
    iter, c, mode)
    eps =0.000001;
    % Update strategy 0 gradient vector
    if mode == 0
        next_b = b + gradient_b / iter;
        next_s = s + gradient_s / iter;
    % Update strategy 1 unit vector
    elseif mode == 1
        mag = sqrt(gradient_b ^ 2 + gradient_s ^ 2);
        
        next_b = b + gradient_b / (mag + eps) * c;
        next_s = s + gradient_s / (mag + eps) * c;
    % Update strategy 2 move in diagonal 
    elseif mode == 2
        if gradient_b > 0
            next_b = b + c;
        else
            next_b = b - c;
        end
        
        if gradient_s > 0
            next_s = s + c;
        else
            next_s = s - c;
        end
    % Update strategy 3 move in largest gradient (eithr b or s)
    elseif mode == 3
        if abs(gradient_s) > abs(gradient_b)
            next_b = b;
            if gradient_s > 0
                next_s = s + c;
            else
                next_s = s - c;
            end
        else
            next_s = s;
            if gradient_b > 0
                next_b = b + c;
            else
                next_b = b - c;
            end
        end
    end
end
