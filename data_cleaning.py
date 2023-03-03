import pandas as pd
from tqdm import tqdm
import time
import sys

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

RETURN_LEAD = 30
CORR_WINDOW = 240
PCT_CHNG_WINDOW = 30
VAR_WINDOW = 240  # consider further increasing the variance window further
RETURN_UPPER_BOUND = 0.90
RETURN_LOWER_BOUND = 0.10


def compile_data(down_file, new_file, cols):
    for year in range(2011, 2022):
        file = pd.read_csv('hist_data/{}{}.csv'.format(down_file, year), delimiter=';',
                           names=cols)
        file['Date Time'] = pd.to_datetime(file['Date Time'], format='%Y-%m-%d %H:%M')
        print(file.head())
        file.to_csv('hist_data/{}_{}adj.csv'.format(new_file, year), index=False)

    file1 = pd.read_csv('hist_data/{}_2011adj.csv'.format(new_file))
    file2 = pd.read_csv('hist_data/{}_2012adj.csv'.format(new_file))
    file3 = pd.read_csv('hist_data/{}_2013adj.csv'.format(new_file))
    file4 = pd.read_csv('hist_data/{}_2014adj.csv'.format(new_file))
    file5 = pd.read_csv('hist_data/{}_2015adj.csv'.format(new_file))
    file6 = pd.read_csv('hist_data/{}_2016adj.csv'.format(new_file))
    file7 = pd.read_csv('hist_data/{}_2017adj.csv'.format(new_file))
    file8 = pd.read_csv('hist_data/{}_2018adj.csv'.format(new_file))
    file9 = pd.read_csv('hist_data/{}_2019adj.csv'.format(new_file))
    file10 = pd.read_csv('hist_data/{}_2020adj.csv'.format(new_file))
    file11 = pd.read_csv('hist_data/{}_2021adj.csv'.format(new_file))

    pd.concat([file1, file2, file3, file4, file5, file6, file7, file8, file9, file10, file11]).to_csv(
        'hist_data/{}comp.csv'.format(new_file), index=False)


def main_data_cleaning():
    wti_oil = pd.read_csv('firstrate/CL_Cont.csv')
    bco_usd = pd.read_csv('hist_data/BCO_USDcomp.csv')
    currencies = pd.read_csv('hist_data/EUR_USDcomp.csv')
    usdjpy = pd.read_csv('hist_data/USD_JPYcomp.csv')
    gbpusd = pd.read_csv('hist_data/GBP_USDcomp.csv')
    usdcad = pd.read_csv('hist_data/USD_CADcomp.csv')
    usdsek = pd.read_csv('hist_data/USD_SEKcomp.csv')
    usdchf = pd.read_csv('hist_data/USD_CHFcomp.csv')

    currs = [usdjpy, gbpusd, usdcad, usdsek, usdchf, bco_usd]
    del_cols = ['High', 'Low', 'Close', 'Volume']

    for cur in tqdm(currs):
        cur.set_index('Date Time', inplace=True)
        currencies = currencies.join(cur, on='Date Time')
    for col in tqdm(currencies):
        if any([remove in col for remove in del_cols]):
            currencies = currencies.drop(col, axis=1)

    currencies['Dollar_Currency'] = 50.14348112 * currencies['Open_Eur'] ** -0.576 * currencies['Open_Jpy'] ** 0.136 * \
                                    currencies['Open_Gbp'] ** -0.119 * currencies['Open_Cad'] ** 0.091 * currencies[
                                        'Open_Sek'] ** 0.042 * currencies['Open_Chf'] ** 0.036

    wti_oil.set_index('Date Time', inplace=True)
    oil_currencies = currencies.join(wti_oil, on='Date Time')

    for col in tqdm(oil_currencies):
        if any([remove in col for remove in del_cols[:3]]):
            oil_currencies = oil_currencies.drop(col, axis=1)

    oil_currencies.dropna(inplace=True)
    oil_currencies = oil_currencies.rename(columns={'Open': 'Open_Wti', 'Volume': 'Volume_Wti'})

    oil_currencies['Spread'] = oil_currencies['Open_Bco'] - oil_currencies['Open_Wti']

    oil_currencies = oil_currencies.reset_index()
    oil_currencies = oil_currencies.drop('index', axis=1)

    inter = oil_currencies[
        ['Open_Eur', 'Open_Jpy', 'Open_Gbp', 'Open_Cad', 'Open_Sek', 'Open_Chf', 'Dollar_Currency', 'Open_Bco',
         'Open_Wti']]

    corr = inter.rolling(CORR_WINDOW).corr()
    pct = inter.pct_change(periods=PCT_CHNG_WINDOW)

    pairs = inter.columns

    check = []

    oil_currencies.dropna(inplace=True)
    print(oil_currencies.head(5))
    sys.exit(0)

    for cola in tqdm(pairs):
        oil_currencies['{}_var'.format(cola)] = oil_currencies[cola].rolling(VAR_WINDOW).var()
        oil_currencies['{}_pct_chng'.format(cola)] = pct[cola]
        # for colb in pairs:
        #     if cola == colb or '{}_{}'.format(cola, colb) in check or '{}_{}'.format(colb, cola) in check:
        #         continue
        #     check.append('{}_{}'.format(cola, colb))
        #     oil_currencies['{}_{}_corr'.format(cola, colb)] = corr.unstack()[cola][colb]




    oil_currencies.dropna(inplace=True)
    oil_currencies = oil_currencies.reset_index()
    oil_currencies = oil_currencies.drop('index', axis=1)

    oil_currencies['Future'] = oil_currencies['Open_Wti'].shift(-RETURN_LEAD)
    oil_currencies['Return'] = (oil_currencies['Future'] - oil_currencies['Open_Wti']) / oil_currencies['Open_Wti']

    def classify(val, b=oil_currencies['Return'].quantile(RETURN_UPPER_BOUND),
                 s=oil_currencies['Return'].quantile(RETURN_LOWER_BOUND)):
        if val >= b:
            return 2
        elif val <= s:
            return 1
        else:
            return 0

    # oil_currencies['target'] = list(map(classify, oil_currencies['Return']))

    oil_currencies = oil_currencies.drop(columns=['Future', 'Return'], axis=1)

    oil_currencies.dropna(inplace=True)
    oil_currencies = oil_currencies.reset_index()
    oil_currencies = oil_currencies.drop('index', axis=1)

    oil_currencies['Date Time'] = pd.to_datetime(oil_currencies['Date Time'], format='%Y.%m.%d %H:%M:%S').map(
        pd.Timestamp.timestamp)
    oil_currencies.dropna(inplace=True)

    # oil_currencies.set_index('Date Time', inplace=True)

    oil_cols = list(oil_currencies.columns)
    end = [oil_cols[0], oil_cols[-1]]
    oil_cols = oil_cols[1:-1] + end
    oil_currencies = oil_currencies[oil_cols]
    oil_currencies.to_csv('loon_01.csv', index=False)


if __name__ == '__main__':
    # compile_data('DAT_ASCII_BCOUSD_M1_', 'BCO_USD',
    #              ['Date Time', "Open_Bco", 'High_Bco', 'Low_Bco', 'Close_Bco', 'Volume_Bco'])
    main_data_cleaning()
    # val = pd.read_csv('loon_01.csv')
    # n = len(val)
    # val = val[:int(n * 0.1)]
    # print(len(val))
    # main_neural()
    # lv = pd.read_csv('loon_01.csv')
    #
    # print(lv['target'].value_counts())



