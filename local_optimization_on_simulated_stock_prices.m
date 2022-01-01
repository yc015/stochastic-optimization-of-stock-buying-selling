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

% Plot one of the simulated stock price
stock_price = stock_prices(:, randi([1, NSample]));

plot(t, stock_price)

hold on

p = [];
p(1) = plot(t, ones(length(t), 1) * exp(2),'g','LineWidth',3, 'DisplayName', "Long-term Mean");
p(2) = plot(t, ones(length(t), 1) * mean(stock_price),'r','LineWidth',3, 'DisplayName', "Stock Price Mean");
title("dX(t) / dt = 0.8(2 - X(t)) + dW(t) / dt",'FontSize', 16)
xlabel("Time", 'FontSize', 16)
ylabel("Price", 'FontSize', 16)
legend(p, 'FontSize', 10)
hold off

b0 = 5;
s0 = 18;
num_iter = 300;

% All this part is about finding the all possible sets of 
% buying and selling times for this given stock using the
% b0 and s0

[buy_times, sell_times] = find_set_of_selling_buying_time(stock_price, b0, s0); 

clf;
% Plot the stock price, buying & selling threshold, and 
% the transactions (shaded period)
plot(t, stock_price)
hold on
plot(t, ones(length(t), 1) * b0)
plot(t, ones(length(t), 1) * s0)


for i=1:length(buy_times)
    one_buy_time = buy_times(i);
    one_sell_time = sell_times(i);
    ix = [one_buy_time: 1: one_sell_time] / (N_T - 1) * T - 1;
    yz = stock_price(one_buy_time:one_sell_time);
    area(ix, yz)
end

legend("Stock Price", "Buying Threshold", "Selling Threshold", "FontSize", 10)
title("All Possible Transactions before Optimizing", "FontSize", 16)
xlabel("Time", 'FontSize', 16)
ylabel("Price", 'FontSize', 16)
hold off

s = [];
b = [];
reward = [];
b(1) = b0;
s(1) = s0;

strategis = ["Adaptive Gradient", "Unit Gradient", "Diagonal", "Uni-directional"];
mode = 1
for iter=1:num_iter
    c = (iter) ^ (-1/2);
    reward(iter) = MultiTransactionReward(stock_price, b(iter), s(iter), T);

    incre_b = b(iter) + c;
    reward_incre_b = MultiTransactionReward(stock_price, incre_b, s(iter), T);

    decre_b = b(iter) - c;
    reward_decre_b = MultiTransactionReward(stock_price, decre_b, s(iter), T);

    incre_s = s(iter) + c;
    reward_incre_s = MultiTransactionReward(stock_price, b(iter), incre_s, T);

    decre_s = s(iter) - c;
    reward_decre_s = MultiTransactionReward(stock_price, b(iter), decre_s, T);

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
    
reward(iter + 1) = MultiTransactionReward(stock_price, b(iter + 1), s(iter + 1), T);
    
figure()
plot(linspace(0, length(reward) - 1, length(reward)), s)
hold on
plot(linspace(0, length(reward) - 1, length(reward)), b)
plot(linspace(0, length(reward) - 1, length(reward)), reward)
legend("Selling Threshold", "Buying Threshold", "Reward" + " " + string(reward(end)), "FontSize", 12)
title("Local Optim with " + strategis(mode + 1), "FontSize", 15)
hold off

[buy_times, sell_times] = find_set_of_selling_buying_time(stock_price, b(end), s(end)); 

clf;
% Plot the stock price, buying & selling threshold, and 
% the transactions (shaded period)
plot(t, ones(length(t), 1) * b(end))
hold on
plot(t, ones(length(t), 1) * s(end))
plot(t, stock_price)

for i=1:length(buy_times)
    one_buy_time = buy_times(i);
    one_sell_time = sell_times(i);
    ix = [one_buy_time: 1: one_sell_time] / (N_T - 1) * T - 1;
    yz = stock_price(one_buy_time:one_sell_time);
    area(ix, yz)
end

legend("Buying Threshold", "Selling Threshold", "Stock Price")
title("All Possible Transactions after Optimizing")
hold off

% Explore the surface of the reward function at all possible b and s
% b and s are the x and y axes of the plot
% and the surface (z) is the reward computed at a given (b, s)
s_all = linspace(1, 26, 50);
b_all = linspace(1, 26, 50);
reward_all = [];
for i=1:length(s_all)
    for j=1:length(b_all)
        reward_all(j, i) = MultiTransactionReward(stock_price, b_all(i), s_all(j), T);
    end
end

clf;
h = surf(b_all, s_all, reward_all)
set(h,'LineStyle','none')
xlabel("Buying Threshold")
ylabel("Selling Threshold")
zlabel("Reward")
%zlim([-50, 50])
hold on

plot3(b(1), s(1), reward(1), 'ogreen','linewidth',3)
plot3(b, s, reward, 'red','linewidth',2)
plot3(b(end), s(end), reward(end), 'oblue','linewidth',6)
text(b(1), s(1), reward(1) + 5, append('Start ', string(reward(1))))
text(b(end), s(end), reward(end) + 5, append('End ', string(reward(end))))
title("Path of Local Optimization", "FontSize", 15)
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

% Local Optimization
% Optimize on multiple transaction of one randomly chosen stock
function reward=MultiTransactionReward(one_stock_price, b, s, T)
    if b > s
        reward = 0;
        return
    end
    [tau_b,ColNrs] = find(one_stock_price < b);
    [tau_s,ColNrs] = find(one_stock_price > s);
    
    buy_times = [];
    sell_times = [];
    
    nth_set = 1;
    prev_s = 1;
    i = 1;
    while i <= length(tau_b)
        b_time = tau_b(i);
        s_time = 0;
        RowNrs = [];
        for j=prev_s:length(tau_s)
            if tau_s(j) > tau_b(i)
                s_time = tau_s(j);
                buy_times(nth_set) = b_time;
                sell_times(nth_set) = s_time;
                nth_set = nth_set + 1;
                prev_s = j + 1;
                
                [RowNrs,ColNrs] = find(tau_b > sell_times(end));
                if length(RowNrs) < 1 
                    break
                end
                if prev_s > length(tau_s)
                    break
                end
                i = RowNrs(1);
                break
            end
        end
        if length(RowNrs) < 1 
            break
        end
        if prev_s > length(tau_s)
            break
        end
        i = i + 1;
    end
    
    tau_b = buy_times;
    tau_s = sell_times;
    
    if length(tau_b) < 1
        reward = 0;
    else
        phi = exp(-0.01 * tau_s / length(one_stock_price) * T) * s * (1 - 0.01) - exp(-0.01 * tau_b / length(one_stock_price) * T) * b * (1 + 0.01);
        reward = sum(phi);
    end
end


function [buy_times, sell_times]=find_set_of_selling_buying_time(one_stock_price, b, s)
    [tau_b,ColNrs] = find(one_stock_price < b);
    [tau_s,ColNrs] = find(one_stock_price > s);
    
    buy_times = [];
    sell_times = [];
    
    nth_set = 1;
    prev_s = 1;
    i = 1;
    while i <= length(tau_b)
        b_time = tau_b(i);
        s_time = 0;
        
        for j=prev_s:length(tau_s)
            if tau_s(j) > tau_b(i)
                s_time = tau_s(j);
                buy_times(nth_set) = b_time;
                sell_times(nth_set) = s_time;
                nth_set = nth_set + 1;
                prev_s = j + 1;
                
                [RowNrs,ColNrs] = find(tau_b > sell_times(end));
                if length(RowNrs) < 1 
                    return
                end
                if prev_s > length(tau_s)
                    return
                end
                i = RowNrs(1);
                break
            end
        end
        i = i + 1;
    end
end
