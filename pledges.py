import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pdb
import argparse
from numba import jit

def destring(x):
    try:
        res = float(x)
    except:
        res = 0.0
    return res

def histo(s, bins=None):
    h = np.histogram(s.dropna(), bins=bins)
    return pd.Series(h[0], index=h[1][0:-1])

def norm(s):
    return s / s.sum()

def reups(colpairrow):
    if colpairrow.iloc[0] > 0:
        # Pledged previous year
        if colpairrow.iloc[1] > 0:
            # And pledged this year
            return 'reup'
        else:
            # Ooops, MIA
            return 'falloff'
    else:
        # No pledge previous year
        if colpairrow.iloc[1] > 0:
            # So this is a new one
            return 'new'
        else:
            # Not new, not a falloff, not a renewal -- irrelevant
            return np.nan

def biggestwo(s):
    news = s.sort_values().dropna()
    return news.iloc[-2:].sum() / news.sum()

def isyear(s):
    return s.apply(lambda x: isinstance(x, int))

def binsums(s, bins=10):
    sums, edges, binnumber = stats.binned_statistic(s, s, statistic='sum', bins=bins)
    return sums


def counts(s, bins=10):
    counts, edges, binnumber = stats.binned_statistic(s, s, statistic='count', bins=bins)
    return counts


def gini(s):
    news = s.sort_values()
    news.dropna(inplace=True)
    index = np.arange(1, news.shape[0] + 1)
    n = news.shape[0]
    return np.sum((2 * index - n - 1) * news) / (n * np.sum(news))


def intervals(s, divisor=5):
    quantiles = [(1.0 / divisor) * x for x in range(1, divisor)]
    q = s.quantile(quantiles)
    return pd.concat([
        pd.Series({0.0: 0.0}),
        q,
        pd.Series({1.0: s.max()})
    ])


def reducehighest(s):
    s.loc[s.idxmax()] = np.nan
    return s


def run(tdf, trim=False):

    if trim:
        tdf = tdf.apply(reducehighest)
        tsuffix = ' -- trimmed'
    else:
        tsuffix = '-- untrimmed'
    maxs = tdf.max()
    print("maxes{}:".format(tsuffix))
    print(maxs)
    medians = tdf.median()
    print("medians{}:".format(tsuffix))
    print(medians)
    sums = tdf.sum()
    print("sums{}:".format(tsuffix))
    print(sums)
    pledges = tdf.count()
    print("pledges{}:".format(tsuffix))
    print(pledges)
    ginis = tdf.apply(gini)
    quantiles = tdf.apply(intervals, divisor=10)
    pctbiggest = (maxs / sums) * 100
    totchanges = sums.pct_change().dropna() * 100




    bs = pd.DataFrame([
        binsums(
            tdf[col],
            bins=quantiles[col]
        )
        for col in tdf.columns
        ],
        index = columns
    )

    bincount = pd.DataFrame([
        counts(
            tdf[col],
            bins=quantiles[col]
        )
        for col in tdf.columns
        ],
        index = columns
    )

    pctbs = bs.apply(
        lambda s : (s / s.sum()) * 100,
        axis=1
    )

    pctcount = bincount.apply(
        lambda s : (s / s.sum()) * 100,
        axis=1
    )

    btwos = tdf.apply(
        biggestwo
    )

    cols = tdf.columns
    tups = [
        (
            cols[ix],
            tdf[
             [cols[ix-1], cols[ix]]]
        )
        for ix in range(1, len(cols))
    ]

    reupdf = pd.DataFrame(
        dict(
            [
                (
                 tup[0],
                 tup[1].apply(
                     reups,
                     axis=1
                    )
                 )
             for tup in tups
            ]
        )
    )

    reuprates = reupdf.apply(lambda x: x.value_counts(normalize=True)) * 100
    ax = reuprates.T.plot(
        kind='bar',
        stacked=True,
        title="New/renewed/fallen off as percent{}".format(tsuffix),
        figsize=(9,5),
        ylim=(0,100)
    )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('pics/reuprates{}.png'.format(tsuffix))
    plt.close('all')

    reupnums = reupdf.apply(lambda x: x.value_counts())
    ax = reupnums.T.plot(
        kind='bar',
        stacked=True,
        title="New/renewed/fallen off, absolute number{}".format(tsuffix),
        figsize=(9,5),
    )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('pics/reupnums{}.png'.format(tsuffix))
    # plt.savefig('pics/reupnums.png')
    plt.close('all')

    reupratios = pd.DataFrame(
        dict(
            [
                (
                 tup[0],
                 tup[1].apply(
                     lambda s : s.iloc[1] / s.iloc[0],
                     axis=1
                    )
                 )
             for tup in tups
            ]
        )
    ).median() * 100 - 100


    reupratios.plot(
        kind='bar',
        legend=False,
        title="Median percent increase/decrease for renewed pledges{}".format(tsuffix),
        ylim=(-10, 10),
        # grid=True
        # logy=True
    )
    # plt.show()
    plt.savefig('pics/reupratios{}.png'.format(tsuffix))
    # plt.savefig('pics/reupratios.png')
    plt.close('all')

    lim = max(abs(totchanges.min()), totchanges.max()) * 1.1

    ax = totchanges.plot(
        kind='bar',
        legend=False,
        title="Percent change in total, year by year{}".format(tsuffix),
        ylim=(lim * -1, lim),
        # grid=True
        # logy=True
    )

    print('Median change in total, year to year{}: {}'.format(
        ', trimmed' if trim else ', untrimmed',
        totchanges.median()
    ))
    # plt.show()
    plt.savefig('pics/totchanges{}.png'.format(tsuffix))
    # plt.savefig('pics/reupratios.png')
    plt.close('all')

    ginis.plot(
        kind='bar',
        legend=False,
        title="Gini ratio{}".format(tsuffix),
        ylim=(0.4, 0.7),
        # logy=True
    )
    # plt.show()
    plt.savefig('pics/gini{}.png'.format(tsuffix))
    # plt.savefig('pics/gini.png')
    plt.close('all')


    pctbs.plot(
        kind='bar',
        stacked=True,
        legend=False,
        title="Percent of total value by decile{}".format(tsuffix)
    )
    # plt.show()
    # plt.savefig('pics/pctofvalue.png')
    plt.savefig('pics/pctofvalue{}.png'.format(tsuffix))
    plt.close('all')

    bincount.plot(
        kind='bar',
        stacked=True,
        legend=False,
        title="Number of pledges by decile{}".format(tsuffix)
    )
    # plt.show()
    plt.savefig('pics/numberbydecile{}.png'.format(tsuffix))
    plt.close('all')

    pctcount.plot(
        kind='bar',
        stacked=True,
        legend=False,
        title="Percent of number of pledges by decile{}".format(tsuffix)
    )
    # plt.show()
    plt.savefig('pics/pctofnumber{}.png'.format(tsuffix))
    # plt.savefig('pics/pctofnumber.png')
    plt.close('all')

    medians.plot(
        kind='bar',
        legend=False,
        title="Median pledge (2017 dollars){}".format(tsuffix),
        # ylim=(0, 2.0 * medians.mean())
        ylim = (0, 3000.0)
    )
    # plt.show()
    plt.savefig('pics/median{}.png'.format(tsuffix))
    # plt.savefig('pics/median.png')
    plt.close('all')

    pctbiggest.plot(kind='bar',
        legend=False,
        title="Biggest pledge as percent of total{}".format(tsuffix),
        ylim=(0,20.0)
    )
    plt.savefig('pics/biggest{}.png'.format(tsuffix))
    # plt.savefig('pics/biggest.png')
    plt.close('all')

    btwos.plot(kind='bar',
        legend=False,
        title="Biggest two pledges as percent of total{}".format(tsuffix),
        ylim=(0, 20.0)
    )
    plt.savefig('pics/biggesttwo{}.png'.format(tsuffix))
    # plt.savefig('pics/biggesttwo.png')
    plt.close('all')

    sumrange = range(0, len(sums))
    slope, intercept, r, p, stderr = stats.linregress(sumrange, sums)
    print(slope, intercept, r, p, stderr)
    linr = pd.Series([intercept + i * slope for i in sumrange], index=sums.index)
    # z = np.polyfit(sumrange, linr.values, 4)
    # f = np.poly1d(z)
    # plt.plot(sumrange, sums.values, 'ro')
    # for x1 in np.linspace(0, sumrange[-1], 50):
    #     plt.plot(x1, f(x1), 'b+')
    # plt.show()
    # Above gives essentially the same result as linregress. In other words, the detail is random.
    # No there there.


    fig, ax  = plt.subplots()
    sums.plot(
        kind='bar',
        legend=False,
        title="Total pledging (2017 dollars){}".format(tsuffix),
        ylim=(0, 700000.0)
    )
    ax2 = ax.twinx()
    ax2.set_ylim(0, 700000.0)
    ax2.tick_params(axis='y', right=False, labelright=False)
    ax2.plot(ax.get_xticks(), linr, color='k')
    # plt.show()
    plt.savefig('pics/totals{}.png'.format(tsuffix))
    plt.close('all')

    bins=np.logspace(1.0, 5.0, num=15)
    # bins=np.logspace(1.0, 12, num=24, base=np.e)
    for year, s in tdf.iteritems():
        syear = str(year)
        ax = s.plot.hist(
            title=syear + tsuffix,
            # bins=15
            bins=bins,
            logx=True,
            ylim=(0,50),
            xticks=[10,100,1000,10000,100000]
        )
        ax.set_ylabel('Number of pledges')
        ax.set_xlabel('Amount of pledge')
        plt.savefig('hists/{}{}.png'.format(
            syear,
            tsuffix
        ))
        plt.close('all')


    valbinned = pd.DataFrame([
        binsums(
            tdf[col],
            bins=bins
        )
        for col in tdf.columns
        ],
        index=tdf.columns,
        columns=bins[0:-1]
    )

    valbinned = valbinned.apply(norm, axis=1) * 100

    for year, s in valbinned.iterrows():
        syear = str(year)
        # ax = s.plot.bar(
        #     title=syear + ' -- value ' + tsuffix,
        #     logx=True,
        #     width=np.diff(bins),
        #     ylim=(0,valbinned.max().max()),
        #     xlim=(1, 120000),
        # )
        # ax.set_ylabel('Total value of pledges')
        # ax.set_xlabel('Amount of pledge')
        fig, ax = plt.subplots()
        ax.bar(
            s.index,
            s.values,
            width=np.diff(bins),
            align='edge'
        )
        ax.set_xscale("log")
        ax.set_ylabel('Percent of total pledged')
        ax.set_xlabel('Amount of pledge')
        ax.set_title(syear + ' -- value ' + tsuffix)
        # ax.set_ylim(0, valbinned.max().max())
        ax.set_ylim(0, 50)
        plt.savefig('valhist/{} -- value {}.png'.format(
            syear,
            tsuffix
        ))
        plt.close('all')

    hdf = tdf.apply(histo, bins=bins)
    hdf.to_csv('pledge_bins{}.csv'.format(
        tsuffix
    ))

    # comps = hdf.sum(axis=1)
    # comps.plot(
    #     # kind='bar',
    #     legend=False,
    #     title="Composite pledging",
    #     logx=True
    # )
    # plt.show()
    # t = tdf.apply(binsums, bins=bins)
    # print(hdf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--debug',
        '-d',
        action='store_true',
        default=False,
        help='Debug mode'
    )
    parser.add_argument(
        '--trim',
        '-t',
        action='store_true',
        default=False,
        help='Remove highest pledge from each year'
    )
    args = parser.parse_args()

    if args.debug:
        pdb.set_trace()

    df = pd.read_excel(
        # 'Y18 Pledges for MJS.xls',
        # sheetname='Y17 & Y18',
        # 'MJS File 12-14-17.xlsx',
        'MJS File 1-29-18.xlsx',
        # 'MJS File 12-14-17-no-outliers.xlsx',
        sheetname='Sheet1',
        header=None
    )

    # These tags need to be put into the spreadsheet, in column A,
    # to show the program where line-item data begins and ends

    hrow = df[df[0] == 'years']
    d0 = df[df[0] == 'data start']
    dn = df[df[0] == 'data end']
    dstartrow = d0.index[0]
    dendrow = dn.index[0] + 1
    columns = hrow.iloc[0, :].where(isyear).dropna()

    df = df.iloc[dstartrow:dendrow, columns.index].fillna(0)
    df.columns = columns
    df.reset_index(drop=True)
    print("totals un-adjusted {}:")
    print(df.sum()),


    df = df.applymap(destring)
    # df.applymap(float)

    inflation = pd.read_excel(
        'inflation.xlsx',
        index_col=0
    )
    avgs = inflation['Avg'].loc[columns.values]
    reflators = avgs[2017] / avgs
    # df = df * reflators
    df = df.multiply(reflators)
    tdf = df.replace(0, np.nan)
    for t in (False, True):
        run(tdf, trim=t)








