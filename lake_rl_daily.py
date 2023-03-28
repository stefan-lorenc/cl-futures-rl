import gym
import numpy as np
from math import floor
import pandas as pd
from gym import spaces
from collections import deque
import itertools
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

ENV_LOOKBACK = 5
TREND = 20

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class Lake(gym.Env):
    def __init__(self):
        super(Lake, self).__init__()
        self.action_space = spaces.Discrete(3)  # long[0], short[1], flat[2]
        # Market Env, Current Position
        oil = pd.read_csv(r'C:\Users\stefa\PycharmProjects\Loon\firstrate\CL_continuous_UNadjusted_1min.txt',
                          names=['Date Time', 'Open_Wti', 'High_Wti', 'Low_Wti', 'Close_Wti', 'Volume'],
                          nrows=10000, skiprows=4825000)

        oil['Date Time'] = pd.to_datetime(oil['Date Time'])
        oil = oil.set_index('Date Time')

        def trend(vals):
            return ((vals.iloc[-1] - vals.iloc[0]) / TREND)

        oil['Trend'] = oil['Close_Wti'].rolling(TREND).apply(trend)

        # oil = oil[['Close_Wti', 'Volume', 'Trend']]
        # oil = oil[oil.index.year == 2020]
        oil = oil[oil.index.dayofweek < 5]
        oil = oil[oil.index.hour >= 9]
        oil = oil[oil.index.hour < 16]
        oil_grouped = oil.groupby(pd.Grouper(freq='D'))

        self.cleaned = [group for _, group in oil_grouped if not group.empty and len(group) >= 380]

        print('Number of available training days ->', len(self.cleaned))

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6 * ENV_LOOKBACK,), dtype=np.float64)

    def step(self, action):

        # self.reward = 0
        prev = self.account_bal

        self.cur_price = self.oil.iloc[self.index]['Close_Wti']

        # record changes to position
        self.open_p_l = self.p_l()

        if self.contracts_avalible_to_trade != 0:
            self.to_open(action)
        elif action == 2 and self.contracts_avalible_to_trade == 0:
            self.to_close()

        self.changes.append((self.account_bal - prev) / prev)

        if self.index % 10 == 0:
            self.reward = self.better_than_nothing()

        # If unable to trade, end the episode
        if floor(self.account_bal / self.margin_rec) - 1 == 0:
            # self.render()
            # self.reward = self.better_than_nothing()
            self.done = True
            print(self.account_bal)

        if self.index == len(self.oil) - 5:
            self.done = True
            self.to_close()
            print(self.account_bal)
            # self.reward = self.better_than_nothing()


        self.update_observation()

        self.observation = np.array(list(itertools.chain(*self.market_env)))



        self.info = {}

        self.index += 1

        return self.observation, self.reward, self.done, self.info

    def reset(self):
        self.done = False
        self.reward = 0

        self.account_bal = 10000
        self.org_acc_bal = self.account_bal
        self.position = 0

        self.margin_rec = 2700  # Initial intraday margin per MCL contract from CME
        self.slippage = 0.01
        self.contracts_avalible_to_trade = floor(self.account_bal / self.margin_rec) - 1

        self.oil = random.choice(self.cleaned)


        self.daily_trades_heading = ['Entry Price', 'Entry Time', 'Contract Quantity', 'Exit Price', 'Exit Time',
                                     'Profit ($)', 'Type (L/S)']
        self.daily_long_trades = pd.DataFrame(columns=self.daily_trades_heading)
        self.daily_short_trades = pd.DataFrame(columns=self.daily_trades_heading)
        self.account_changes = pd.DataFrame(columns=['Account Balance'])
        self.changes = []
        self.trade_entry_time = None

        self.index = ENV_LOOKBACK
        self.position_price = -1
        self.cur_price = -1
        self.contracts_long, self.contracts_short = 0, 0
        self.open_p_l = 0

        self.market_env = deque(maxlen=ENV_LOOKBACK)
        self.index = ENV_LOOKBACK

        for ind in range(ENV_LOOKBACK):

            initialize = self.oil.iloc[[ind]].values.flatten().tolist()
            initialize = list(initialize[i] for i in [3, 4, 5])
            initialize.extend([self.position, self.open_p_l, self.account_bal - self.org_acc_bal])
            self.market_env.append(initialize)

            acc_change = pd.Series({'Account Balance': self.account_bal})
            self.account_changes = pd.concat([self.account_changes, acc_change.to_frame().T], ignore_index=True)

        self.observation = np.array(list(itertools.chain(*self.market_env)))


        return self.observation

    def to_open(self, action):

        if action == 0:  # checks for an existing position
            self.contracts_long = self.contracts_avalible_to_trade
            self.account_bal -= (0.25 * self.contracts_long)
            self.position_price = self.cur_price + self.slippage
            self.position = 1

        elif action == 1:
            self.contracts_short = self.contracts_avalible_to_trade
            self.account_bal -= (0.25 * self.contracts_short)
            self.position_price = self.cur_price - self.slippage
            self.position = -1


        self.trade_entry_time = self.oil.iloc[self.index].name
        self.contracts_avalible_to_trade = 0

    def to_close(self):
        position = self.contracts_long + self.contracts_short

        if self.contracts_long > 0:

            new_trade = pd.Series({'Entry Price': self.position_price, 'Entry Time': self.trade_entry_time,
                                   'Contract Quantity': self.contracts_long, 'Exit Price': self.cur_price,
                                   'Exit Time': self.oil.iloc[self.index].name, 'Profit ($)': self.open_p_l,
                                   'Type (L/S)': 'Long'})

            self.daily_long_trades = pd.concat([self.daily_long_trades, new_trade.to_frame().T], ignore_index=True)

        elif self.contracts_short > 0:
            new_trade = pd.Series({'Entry Price': self.position_price, 'Entry Time': self.trade_entry_time,
                                   'Contract Quantity': self.contracts_short, 'Exit Price': self.cur_price,
                                   'Exit Time': self.oil.iloc[self.index].name, 'Profit ($)': self.open_p_l,
                                   'Type (L/S)': 'Short'})

            self.daily_short_trades = pd.concat([self.daily_short_trades, new_trade.to_frame().T], ignore_index=True)

        self.account_bal -= (0.25 * position)
        self.contracts_long, self.contracts_short = 0, 0
        self.account_bal += self.open_p_l - (position * self.slippage * 100)


        self.contracts_avalible_to_trade = floor(self.account_bal / self.margin_rec) - 1
        self.open_p_l = 0
        self.position_price = -1
        self.position = 0

        self.trade_entry_time = None

    def p_l(self):
        live_contracts = self.contracts_long + self.contracts_short
        interim_return = (live_contracts * (self.cur_price - self.position_price)) * 100
        if self.contracts_short > 0:
            interim_return = interim_return * -1

        return interim_return

    def render(self, mode='human', close=False):
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.7, 0.15, 0.15])

        fig.add_trace(go.Candlestick(x=self.oil.index, open=self.oil['Open_Wti'], high=self.oil['High_Wti'],
                                     low=self.oil['Low_Wti'], close=self.oil['Close_Wti'],
                                     increasing_line_color='rgba(107, 107, 107, 0.8)',
                                     decreasing_line_color='rgba(210, 210, 210, 0.8)', name='WTI Oil OHLC'), row=1,
                      col=1)

        fig.add_trace(go.Scatter(x=self.daily_long_trades['Entry Time'], y=self.daily_long_trades['Entry Price'],
                                 customdata=self.daily_long_trades, mode='markers', marker_symbol='triangle-up-dot',
                                 marker_size=13,
                                 marker_line_width=2, marker_line_color='rgba(0, 0, 0, 0.7)',
                                 marker_color='rgba(0, 255, 0, 0.7)',
                                 hovertemplate='Entry Price: %{y:.2f}<br>' +
                                               'Entry Time: %{customdata[1]}<br>' +
                                               'Contract Quantity: %{customdata[2]}<br>' +
                                               'Exit Price: %{customdata[3]}<br>' +
                                               'Exit Time: %{customdata[4]}<br>' +
                                               'Profit ($): %{customdata[5]}<br>' +
                                               'Type (L/S): %{customdata[6]}', name='Long Entries'), row=1, col=1)

        fig.add_trace(go.Scatter(x=self.daily_short_trades['Entry Time'], y=self.daily_short_trades['Entry Price'],
                                 customdata=self.daily_short_trades, mode='markers', marker_symbol='triangle-down-dot',
                                 marker_size=13,
                                 marker_line_width=2, marker_line_color='rgba(0, 0, 0, 0.7)',
                                 marker_color='rgba(255, 0, 0, 0.7)',
                                 hovertemplate='Entry Price: %{y:.2f}<br>' +
                                               'Entry Time: %{customdata[1]}<br>' +
                                               'Contract Quantity: %{customdata[2]}<br>' +
                                               'Exit Price: %{customdata[3]}<br>' +
                                               'Exit Time: %{customdata[4]}<br>' +
                                               'Profit ($): %{customdata[5]}<br>' +
                                               'Type (L/S): %{customdata[6]}', name='Short Entries'), row=1, col=1)

        exits = pd.concat([self.daily_long_trades, self.daily_short_trades], ignore_index=True)
        exits = exits.sort_values(by=['Exit Time'])

        fig.add_trace(go.Scatter(x=exits['Exit Time'], y=exits['Exit Price'],
                                 customdata=exits, mode='markers', marker_symbol='diamond-dot', marker_size=13,
                                 marker_line_width=2, marker_line_color='rgba(0, 0, 0, 0.7)',
                                 marker_color='rgba(173, 173, 168, 0.7)',
                                 hovertemplate='Exit Price: %{y:.2f}<br>' +
                                               'Exit Time: %{customdata[4]}<br>' +
                                               'Contract Quantity: %{customdata[2]}<br>' +
                                               'Profit ($): %{customdata[5]}<br>' +
                                               'Type (L/S): %{customdata[6]}', name='Exits'), row=1, col=1)

        fig.add_trace(
            go.Scatter(x=self.oil.index, y=self.oil['Volume'], marker_color='rgba(126, 189, 168, 0.7)', name='Volume'),
            row=2,
            col=1)

        fig.add_trace(go.Scatter(x=self.oil.index, y=self.account_changes['Account Balance'],
                                 marker_color='rgba(235, 115, 115, 0.7)', name='Account Balance'),
                      row=3,
                      col=1)

        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.show()

    def update_observation(self):

        initialize = self.oil.iloc[[self.index]].values.flatten().tolist()
        initialize = list(initialize[i] for i in [3, 4, 5])

        initialize.extend([self.position, self.open_p_l, self.account_bal - self.org_acc_bal])
        self.market_env.append(initialize)

        acc_change = pd.Series({'Account Balance': self.account_bal})
        self.account_changes = pd.concat([self.account_changes, acc_change.to_frame().T], ignore_index=True)

    def better_than_nothing(self):
        # calculate the reward based on the change in balance
        # reward = (self.account_bal - self.org_acc_bal) / self.org_acc_bal
        reward = ((self.account_bal + self.open_p_l) - self.org_acc_bal) / self.org_acc_bal

        # calculate the Sortino ratio based on the returns
        sortino_ratio = self._calculate_sortino_ratio()

        # adjust the reward based on the Sortino ratio

        if reward < 0 and sortino_ratio < 0:
            reward *= 0.5
        else:
            reward *= sortino_ratio

        # min_reward = -10
        # reward = max(reward, min_reward)

        return reward

    def _calculate_sortino_ratio(self):
        returns = np.array(self.changes)
        downside_returns = np.minimum(returns, 0)
        downside_std = np.std(downside_returns)
        sortino_ratio = np.mean(returns) / downside_std if downside_std > 0 else 0
        return sortino_ratio


