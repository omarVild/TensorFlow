import pandas as pd
import matplotlib.pyplot as plt


fruits = pd.read_csv('fruits.csv')

print fruits.head()
weight_F_TMP= fruits.iloc[1:,0]
colour_F_TMP= fruits.iloc[1:,1]


applesTMP = fruits.query("(Label == 'Apple')")
weightATMP= applesTMP.iloc[1:,0]
colourATMP= applesTMP.iloc[1:,1]


bananaTMP = fruits.query("(Label == 'Banana')")
weightBTMP= bananaTMP.iloc[1:,0]
colourBTMP= bananaTMP.iloc[1:,1]


plt.plot(weightATMP, colourATMP, 'ro', label='Apple')
plt.plot(weightBTMP, colourBTMP, 'yo', label='Bananas')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)


plt.show()