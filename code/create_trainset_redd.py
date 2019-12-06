from redd_parameters import *
import pandas as pd
import time


DATA_DIRECTORY = 'data/REDD/low_freq/'
SAVE_PATH = 'created_data/'


def main():
    start_time = time.time()
    sample_seconds = 8
    validation_percent = 10
    nrows = None
    # I set debug to True to see what's going on.
    debug = True
    appliance_name = 'microwave'
    print('==> The appliance name is {}.'.format(appliance_name))
    train = pd.DataFrame(columns=['aggregate', appliance_name])

    for h in params_appliance[appliance_name]['houses']:
        print('    ' + DATA_DIRECTORY + 'house_' + str(h) + '/'
              + 'channel_' +
              str(params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(h)]) +
              '.dat')

        # read data
        mains1_df = pd.read_table(DATA_DIRECTORY + '/' + 'house_' + str(h) + '/' + 'channel_' +
                                  str(1) + '.dat',
                                  sep="\s+",
                                  nrows=nrows,
                                  usecols=[0, 1],
                                  names=['time', 'mains1'],
                                  dtype={'time': str},
                                  )

        mains2_df = pd.read_table(DATA_DIRECTORY + '/' + 'house_' + str(h) + '/' + 'channel_' +
                                  str(2) + '.dat',
                                  sep="\s+",
                                  nrows=nrows,
                                  usecols=[0, 1],
                                  names=['time', 'mains2'],
                                  dtype={'time': str},
                                  )
        app_df = pd.read_table(DATA_DIRECTORY + '/' + 'house_' + str(h) + '/' + 'channel_' +
                               str(params_appliance[appliance_name]['channels']
                                   [params_appliance[appliance_name]['houses'].index(h)]) + '.dat',
                               sep="\s+",
                               nrows=nrows,
                               usecols=[0, 1],
                               names=['time', appliance_name],
                               dtype={'time': str},
                               )

        mains1_df['time'] = pd.to_datetime(mains1_df['time'], unit='s')
        mains2_df['time'] = pd.to_datetime(mains2_df['time'], unit='s')

        mains1_df.set_index('time', inplace=True)
        mains2_df.set_index('time', inplace=True)

        mains_df = mains1_df.join(mains2_df, how='outer')

        mains_df['aggregate'] = mains_df.iloc[:].sum(axis=1)
        # resample = mains_df.resample(str(sample_seconds) + 'S').mean()

        mains_df.reset_index(inplace=True)

        # deleting original separate mains
        del mains_df['mains1'], mains_df['mains2']

        if debug:
            print("    mains_df:")
            print(mains_df.head())
            # plt.plot(mains_df['time'], mains_df['aggregate'])
            # plt.show()

            # Appliance
            # app_df = app_df.set_index(app_df.columns[0])
            # app_df.index = pd.to_datetime(app_df.index, unit='s')
        app_df['time'] = pd.to_datetime(app_df['time'], unit='s')
        # app_df.columns = [appliance_name]
        if debug:
            print("app_df:")
            print(app_df.head())
            # plt.plot(app_df['time'], app_df[appliance_name])
            # plt.show()

            # the timestamps of mains and appliance are not the same, we need to align them
            # 1. join the aggragte and appliance dataframes;
            # 2. interpolate the missing values;
        mains_df.set_index('time', inplace=True)
        app_df.set_index('time', inplace=True)

        df_align = mains_df.join(app_df, how='outer'). \
            resample(str(sample_seconds) +
                     'S').mean().fillna(method='backfill', limit=1)
        df_align = df_align.dropna()

        df_align.reset_index(inplace=True)
        # print(df_align.count())
        # df_align['OVER 5 MINS'] = (df_align['time'].diff()).dt.seconds > 9
        # df_align.plot()
        # plt.plot(df_align['OVER 5 MINS'])
        # plt.show()

        del mains1_df, mains2_df, mains_df, app_df, df_align['time']

        mains = df_align['aggregate'].values
        app_data = df_align[appliance_name].values
        # plt.plot(np.arange(0, len(mains)), mains, app_data)
        # plt.show()

        if debug:
                # plot the dtaset
            print("df_align:")
            print(df_align.head())
            # plt.plot(df_align['aggregate'].values)
            # plt.plot(df_align[appliance_name].values)
            # plt.show()

        if h == params_appliance[appliance_name]['test_build']:
            # Test CSV
            print(SAVE_PATH + appliance_name +
                  '_test_.pkl.zip')
            df_align.to_pickle(SAVE_PATH + appliance_name +
                               '_test_.pkl.zip', compression='zip')
            print("    Size of test set is {:.4f} M rows.".format(
                len(df_align) / 10 ** 6))
            continue

        train = train.append(df_align, ignore_index=True)
        del df_align

        # Validation CSV
    val_len = int((len(train)/100)*validation_percent)
    val = train.tail(val_len)
    val.reset_index(drop=True, inplace=True)
    train.drop(train.index[-val_len:], inplace=True)
    val.to_pickle(SAVE_PATH + appliance_name + '_validation_' +
                  '.pkl.zip', compression='zip')

    # Training CSV
    train.to_pickle(SAVE_PATH + appliance_name +
                    '_training_.pkl.zip', compression='zip')

    print("    Size of total training set is {:.4f} M rows.".format(
        len(train) / 10 ** 6))
    print("    Size of total validation set is {:.4f} M rows.".format(
        len(val) / 10 ** 6))
    del train, val

    print("\nPlease find files in: " + SAVE_PATH)
    # tot = int(int(time.time() - start_time) / 60)
    print("Total elapsed time: {:.2f} min.".format(
        (time.time() - start_time) / 60))


if __name__ == '__main__':
    main()
