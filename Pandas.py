import pandas as pd
import numpy as np
s = [0, 0, 1, 2, 3, 4, 5]
dates = pd.date_range('1/1/2020', periods=8)
df = pd.DataFrame(np.random.randn(8, 4), index=dates,
                  columns=['A', 'B', 'C', 'D'])
df[['A', 'B']]
df.loc[:, ['B', 'A']]
# ! Attribute access
sa = pd.Series([1, 2, 3], index=list('abc'))
dfa = df.copy()
sa.b
dfa.A
dfa.A = list(range(len(dfa.index)))
type(sa)
dfa['A'] = list(range(len(dfa.index)))
dfa
x = pd.DataFrame({'x': [1, 2, 3], 'y': [3, 4, 5]})
x.loc[1] = {'x': 9, 'y': 99}
x
# ! Selection by label
df1 = pd.DataFrame(np.random.randn(5, 4), columns=list(
    'ABCD'), index=pd.date_range('20200101', periods=5))
df1.loc[2:3]
df1.loc['20200102':'20200104']
s1 = pd.Series(np.random.randn(6), index=list('abcdef'))
s1.loc['c':]
s1.loc['b']
df1 = pd.DataFrame(np.random.randn(6, 4), index=list(
    'abcdef'), columns=list('ABCD'))
df1.loc[['a', 'b', 'c'], :]
df1.loc['d':, 'A':'C']
df1.loc[:, 'A']
df1.loc['a'] > 0
df1.loc[:, df1.loc['a'] > 0]
# * Slicing with labels
s = pd.Series(list('abcde'), index=[0, 3, 2, 5, 4])
s.loc[3:5]
s.sort_index()
s.sort_index().loc[1:6]
s.iloc[1:3]

# * Selection by Position
# order to get purely integer based indexing

s1 = pd.Series(np.random.randn(5), index=list(range(0, 10, 2)))
s1.iloc[:3]
df2 = pd.DataFrame(np.random.randn(6, 4), index=list(
    range(0, 12, 2)), columns=list(range(0, 8, 2)))
df2.iloc[:3, :3]
df2.iloc[[1, 3, 5], [1, 3]]
df2
df2.iloc[1:3, :]
df2.iloc[:, 1:3]
df2.iloc[1, 1]
df2.iloc[1]
x = list('abcdef')
x
x[4:10]
s = pd.Series(x)
s
s.iloc[4:10]
s.iloc[8:10]

df2.iloc[[4, 5, 6]]  # ! Error

# * Selection by callable
df1 = pd.DataFrame(np.random.randn(6, 4), index=list(
    'abcdef'), columns=list('ABCD'))
df1.loc[lambda df:df['A'] > 0, :]
df1
df1.loc[:, lambda df: ['A', 'B']]
df1.iloc[:, lambda df:[0, 1]]
df1[lambda df:df.columns[0]]
df1['A'].loc[lambda s:s > 0]

bb = pd.read_csv('baseball.csv')
bb.columns
bb.head()
(bb.groupby(['Year', 'Team']).sum().loc[lambda df:df['Playoffs'] > 0])
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=list('abc'))
df.ix[[0, 2], 'A']
df.iloc[[0, 2], df.columns.get_indexer(['A', 'B'])]
df.loc[df.index[[0, 2]], ['A', 'B']]

# * Indexing with list missing labels is deprecated

s = pd.Series([1, 2, 3])
s
s.loc[[0, 2]]

# ? Reindexing
s.reindex([1, 2, 3])
s
labels = [1, 2, 3]
s.loc[s.index.intersection(labels)]

s = pd.Series(np.arange(4), index=['a', 'b', 'c', 'd'])
labels = ['c', 'd']
s.reindex(labels)
s

# * Selecting random samples
s = pd.Series([0, 1, 2, 3, 4, 5])
s.sample(n=3)
s.sample(frac=0.5)

s.sample(n=6, replace=False)
s.sample(n=6, replace=True)
df2 = pd.DataFrame({'col1': [9, 8, 7, 6], 'weight_column': [0.5, 0.4, 0.1, 0]})
df2.sample(n=3, weights='weight_column')

df2.sample(n=1, axis=1)
# ? seed for sample random number generator using the random state
# TODO: With the given seed, the sample will always draw the same rows.
df2.sample(n=2, random_state=2)
##################Day-2######################
# * Fast scalar value getting and setting

s = pd.Series([1, 2, 3, 4, 5])
s.iat[3]
df.iat[1, 0]
df

#! at and iat is simmilar as loc and iloc in use

# * Boolean indexing
s = pd.Series(range(-3, 4))
s[s > 0]
s[(s < -1) | (s > 0.5)]
s[~(s < 0)]
df[df['A'] > 1]
df

df2 = pd.DataFrame({'a': ['one', 'one', 'two', 'three', 'two', 'one', 'six'],
                    'b': ['x', 'y', 'y', 'x', 'y', 'x', 'x'],
                    'c': np.random.randn(7)})

criterion = df2['a'].map(lambda x: x.startswith('t'))
df2[criterion]
# ? below equivalent but slower than map function
df2[[x.startswith('t') for x in df2['a']]]

# ? Multiple criteria
df2[criterion & (df2['b'] == 'x')]

df2.loc[criterion & (df2['b'] == 'x'), 'b':'c']

# * Indexing with isin
s = pd.Series(np.arange(5), index=np.arange(5)[::-1], dtype='int64')
s
s.isin([2, 4, 6])
s[s.isin([2, 4, 6])]
s[s.index.isin([2, 4, 6])]
s
s.reindex([2, 4, 6])
df = pd.DataFrame({'vals': [1, 2, 3, 4], 'ids': [
                  'a', 'b', 'f', 'n'], 'ids2': ['a', 'n', 'c', 'n']})
values = ['a', 'b', 1, 3]
df.isin(values)

values = {'ids': ['a', 'b'], 'vals': [1, 3]}
values
df.isin(values)

# * The Where method and Masking

s[s > 0]
# TODO:it can be represented as follows ,To return a Series of the same shape as the original
s.where(s > 0)
df.where(df < 0, -df)  # work on integers "<0"
df
s2 = s.copy()
s2[s2 < 0] = 0
s2
df2 = df.copy()
df2[df2['vals'] < 2] = 0
df2

df = pd.DataFrame(np.random.randn(5, 4), columns=list(
    'ABCD'), index=pd.date_range('20200101', periods=5))
df_orig = df.copy()
df
df_orig.where(df > 0, -df, inplace=True)
df_orig

# * Difference between numpy.where() and DataFrame.where()
df.where(df < 0, -df) == np.where(df < 0, df, -df)

df2 = df.copy()
df2[df2[1:4] > 0] = 3
df2.where(df2 > 0, df2['A'], axis='index')
df2 = df.copy()
df.apply(lambda x, y: x.where(x > 0, y), y=df['A'])

df3 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
df3.where(lambda x: x > 4, lambda x: x+10)

# TODO: find the difference between DataFrame.where() and numpy.where()
# mask is opposite to where
s.mask(s >= 0)
df.mask(df >= 0)


#! IMPORTANT The Query Method

n = 10
df = pd.DataFrame(np.random.randn(n, 3), columns=list('abc'))
df
# pure python

df[(df['a'] < df['b']) & (df['b'] < df['c'])]

# query
df.query('(a<b) & (b<c)')
df = pd.DataFrame(np.random.randint(n/2, size=(n, 2)), columns=list('bc'))
df.index.name = 'a'
# TODO: Difference above line and below line
df.query('a > b and b > c')

df = pd.DataFrame(np.random.randint(n, size=(n, 2)), columns=list('bc'))
df
df.query('index < b < c')
df.index.name = 'a'
df = pd.DataFrame({'a': np.random.randint(5, size=5)})
df.index.name = 'a'
df.query('a > 2')
df.query('index > 2')

# * MUltiIndex query() Syntax

colors = np.random.choice(['red', 'green'], size=n)
foods = np.random.choice(['eggs', 'ham'], size=n)
colors
foods


index = pd.MultiIndex.from_arrays([colors, foods], names=['color', 'food'])
df = pd.DataFrame(np.random.randn(n, 2), index=index)
df.query('color =="red"')

df.query('ilevel_0 == "red"')

##############################Day-3######################
df = pd.DataFrame(np.random.randn(n, 3), columns=list('abc'))
df.index.name = "index"
df2 = pd.DataFrame(np.random.randn(n+2, 3), columns=df.columns)
df2
np.random.randn(1, 3)
expr = '0.0 <= a <= c <= 0.5'
map(lambda frame: frame.query(expr), [df, df2])
df.query('(a<b) &(b<c)')
df[(df['a'] < df['b']) & (df['b'] < df['c'])]
# * in can also be use in
df.query('a in b and c<a')

# * Pure python

df[df['b'] . isin(df['a']) & (df['c'] < df['a'])]

df[df['b'].isin(df['a'])]
df[3] = -0.548771
df

# TODO: how to change a  Particular value at the location of DataFrame


# ? If there  is an duplicate items then remove them just by applying two of the below methods
# ? 1. duplicated() 2.drop_duplicates()
df2.duplicated('a')
df2.duplicated('a', keep='last')
df2.index.duplicated()

# *Dictionary-like get() methods

s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
s.get('a')  # ! equivalent to s['a']
s.get('x', default=-1)

# important: Sometime you want to extract set of values given sequence of rows labels and column labels,and lookup method help with this
dflookup = pd.DataFrame(np.random.rand(20, 4), columns=['A', 'B', 'C', 'D'])
dflookup
dflookup.lookup(list(range(0, 10, 2)), ['B', 'C', 'A', 'B', 'D'])

index = pd.Index(['e', 'd', 'a', 'b'])
index
'd' in index
index = pd.Index(['e', 'd', 'a', 'b'], name='something')
index.names
index = pd.Index(list(range(5)), name='rows')
columns = pd.Index(['A', 'B', 'C'], name='cols')
df = pd.DataFrame(np.random.randn(5, 3), index=index, columns=columns)
df
df['A']
data = pd.DataFrame([list('abcd'), list('efgh'), list('ijkl'), list(
    'mnop')], columns=pd.MultiIndex.from_product([['one', 'two'], ['first', 'second']]))
data.loc[:, ('one', 'second')]

# $ Creating a MultiIndexing objects



# * Sorting a Multiindex

#################### important: Day -4 ############################
# * Take Methods replacement to iloc
import numpy as np
import pandas as pd
index = pd.Index(np.random.randint(0,1000,10))
positions = [0,9,3]
index[positions]
index.take(positions)
ser = pd.Series(np.random.randn(10))
ser.iloc[positions]
ser.take(positions)

frm = pd.DataFrame(np.random.randn(5, 3))
frm.take([1,4,3])
frm.take([0,2],axis=1)

arr = np.random.randn(10)
arr[[0,1]]
arr.take([False,False,True,True])

#? Checking the Performance takes can handle only small range of indexes

arr = np.random.randn(10000,5)
indexer = np.arange(10000)
random.shuffle(indexer)
len(arr)
arr.ndim

#important: but majjor of time for long array iloc is better than take generaally for big amount of data

# ! Interval Index
# note: This can be used in Series and Dataframes
#####################* Day-5#############
import pandas as pd
df = pd.DataFrame({'A':[1,2,3,4,5]},index= pd.IntervalIndex.from_breaks([0,1,2,3,4,5]))
df
df.loc[2]
df.loc[[2,3]]
df.loc[2.5]
df.loc[[2.5,3.5]]
df.loc[pd.Interval(1,2)]


#! Merge,join,and Concatenate

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
 'B': ['B0', 'B1', 'B2', 'B3'],
 'C': ['C0', 'C1', 'C2', 'C3'],
 'D': ['D0', 'D1', 'D2', 'D3']},
 index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
 'B': ['B4', 'B5', 'B6', 'B7'],
 'C': ['C4', 'C5', 'C6', 'C7'],
 'D': ['D4', 'D5', 'D6', 'D7']},
 index=[4, 5, 6, 7])

 df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
 'B': ['B8', 'B9', 'B10', 'B11'],
 'C': ['C8', 'C9', 'C10', 'C11'],
 'D': ['D8', 'D9', 'D10', 'D11']},
 index=[8, 9, 10, 11])

 result = pd.concat(frames,keys=['x','y','z','a'])
 result.loc['y']
 result = pd.concat([df1,df3],axis=1,join="inner")
 result

 frames = [df1,df2,df3]
 result = pd.concat(frames)
 
 # note: concat and append() are work mostly simmilar

 result = df1.append(df2)
 result
s1 = pd.Series(['X0', 'X1', 'X2', 'X3'], name='X')
result = pd.concat([df1,s1],axis=1)
s3 = pd.Series([0,1,2,3],name="foo")
s4 = pd.Series([0,1,2,3])
s5 = pd.Series([0,1,4,5])

pd.concat([s3,s4,s5],axis=1)
pd.concat([s3,s4,s5],axis=1,keys=["A","B","C"])

#note: you can also concat dict 
piece  = {'x':df1,'y':df2,'z':df3}
pd.concat(piece)

#! Merging
left = pd.DataFrame({'A': [1, 2], 'B': [2, 2]})
right = pd.DataFrame({'A': [4, 5, 6], 'B': [2, 2, 2]})
result  = pd.merge(left,right,on='B',how='outer')
result
# ! joining on index
Dataframe.join()
left = pd.DataFrame({'A':['A0','A1'],'B':['B0','B1']},index=['K0','K1'])
right = pd.DataFrame({'C':['C0','C1'],'D':['D0','D1']},index=['K0','K1'])
result = left.join(right)
result
result = left.join(right,how='outer')

#$ Now its time to reshape data and pivot tables
#! reshaping by pivotting dataframe objects
import numpy as np
df = pd.DataFrame({'date':pd.date_range('01/01/2020',periods=8),'value':np.random.randn(8),'variable':['A','A','B','B','C','C','D','D']})
df[df['variable']=='A']
df.pivot(index='variable',columns='value',values='date')
#note: pivot help in reshaping the data into your desired form

#! Reshaping by stacking and unstacking
tuples = list(zip(*[['bar','bar','bar','foo','foo'],['one','two','three','four','five']]))

index = pd.MultiIndex.from_tuples(tuples,names=['first','second'])
df = pd.DataFrame(np.random.randn(5,2),index=index,columns=['A','B'])
df2 = df[:4]
df
stacked = df2.stack()
stacked