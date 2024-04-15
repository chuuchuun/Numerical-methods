import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("7203_jp_d.csv")

y = data['Zamkniecie'].to_numpy()
x = np.arange(1, 1001)


def calculate_ema(prices, N, curr):
    alpha = 2 / (N + 1)
    base = 1 - alpha
    numerator = 0
    denominator = 0
    actual_range = min(curr, N - 1)

    for i in range(actual_range + 1):
        x = pow(base, i)
        denominator += x
        if curr - i > -1:
            x *= prices[curr - i]
            numerator += x

    if denominator != 0:
        ema = numerator / denominator
    else:
        ema = prices[0]
    return ema


def calculate_macd(prices, curr):
    ema12 = calculate_ema(prices, 12, curr)
    ema26 = calculate_ema(prices, 26, curr)
    return ema12 - ema26



macd = [calculate_macd(y, i) for i in range(len(y))]
signal = []

for i in range(len(y)):
    signal_value = calculate_ema(macd, 9, i)
    signal.append(signal_value)

plt.figure(figsize=(14, 7))
plt.plot(x, y, label='Cena Zamknięcia')
plt.title('Cena zamkniecia Toyota Motor Co. (7203.JP) w ciagu ostatnich 1000 dni')
plt.legend()
plt.savefig("1.png")
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(x, macd, label='MACD')
plt.plot(x, signal, label='Linia sygnału')
plt.title('MACD i Linia Sygnału')
plt.legend()
plt.show()

buy_point = []
sell_point = []
buy_value = []
sell_value = []
num_shares = 1000.0
budget = 0.0
transactions = []
statistics = []
for i in range(len(x) - 1):
    if macd[i] < signal[i] and macd[i + 1] > signal[i + 1]:
        if budget >= y[i + 1] * 0.01:
            num_shares = budget / y[i + 1]
            num_shares = round(num_shares, 2)
            budget -= num_shares * y[i + 1]
            buy_point.append((i + 1, signal[i + 1]))
            buy_value.append((i + 1, y[i + 1]))
            transactions.append(('buy', i + 1, y[i + 1], num_shares))
    elif macd[i] > signal[i] and macd[i + 1] < signal[i + 1]:
        if num_shares > 0:
            budget += num_shares * y[i + 1]
            sell_point.append((i + 1, signal[i + 1]))
            sell_value.append((i+1, y[i + 1]))
            transactions.append(('sell', i + 1, y[i + 1], num_shares))
if num_shares > 0:
    budget += num_shares * y[999]
    sell_point.append((1000, signal[999]))
    sell_value.append((1000, y[999]))
    transactions.append(('sell', len(x), y[-1], num_shares))
    num_shares = 0
for i in range(0, len(transactions) - 1, 2):
    sell = transactions[i]
    buy = transactions[i + 1] if i + 1 < len(transactions) else None
    if buy and sell and buy[0] == 'buy' and sell[0] == 'sell':
        profit_loss = (sell[2] - buy[2]) * sell[3]
        statistics.append({
            'buy_index': buy[1],
            'sell_index': sell[1],
            'buy_price': buy[2],
            'sell_price': sell[2],
            'num_shares': sell[3],
            'profit_loss': profit_loss
        })
losses = 0
earnings = 0
for stat in statistics:
    if stat['profit_loss'] > 0:
        earnings += 1
    else:
        losses += 1
print(f"Losses: {losses} Earnings: {earnings}")
# Print detailed statistics for each transaction
for stat in statistics:
    print(f"Transaction from day {stat['sell_index']} to {stat['buy_index']}:")
    print(f"  Buy Price: {stat['buy_price']}, Sell Price: {stat['sell_price']}, Num Shares: {stat['num_shares']}")
    print(f"  Profit/Loss: {stat['profit_loss']}")

print(f'End budget: {budget}')
print(f'End number of shares: {num_shares}')
buy_point_x, buy_point_y = zip(*buy_point) if buy_point else ([], [])
sell_point_x, sell_point_y = zip(*sell_point) if sell_point else ([], [])

buy_value_x, buy_value_y = zip(*buy_value) if buy_value else ([], [])
sell_value_x, sell_value_y = zip(*sell_value) if sell_value else ([], [])

plt.figure(figsize=(14, 7))
plt.plot(x, macd, label='MACD')
plt.plot(x, signal, label='Linia sygnału')
if buy_point:
    plt.scatter(buy_point_x, buy_point_y, color='green', label="Punkty kupna", marker='o')
if sell_point:
    plt.scatter(sell_point_x, sell_point_y, color='red', label="Punkty sprzedaży", marker='o')
plt.title('MACD i Linia Sygnału z Punktami Transakcji')
plt.legend()
plt.savefig("2.png")
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(x, y, label='Cena Zamknięcia')
if buy_value:
    plt.scatter(buy_value_x, buy_value_y, color='green', label="Punkty kupna", marker='o')
if sell_value:
    plt.scatter(sell_value_x, sell_value_y, color='red', label="Punkty sprzedaży", marker='o')
plt.title('Cena z Punktami Kupna i Sprzedaży')
plt.legend()
plt.savefig("3.png")
plt.show()
