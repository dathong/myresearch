import re
import pandas as pd
import utils

for m in re.compile("10(1)+01").finditer("10010101001"):
    print(m.start(),m.group())

print(re.search("10000001", "10010101001"))

df = pd.read_csv('long_states_df.csv',nrows=15000)

states_df = utils.process_df1(df['states'])
states_lg = utils.process_df1(df['logit_seq'])
digit_seq = df['words'].values

maxCount = 0
count = 0
for i in range(0,12000,15):

    count = 0
    currCount = 0
    for x in digit_seq[i:i + 15]:
        if x == 1:
            currCount+=1
            if currCount > count:
                count = currCount
        else:
            currCount = 0


    # print('count = ',count)
    if count > 6:
        print('seq = ', digit_seq[i:i + 15])
    if count > maxCount:
        maxCount = count

print('maxCount = ',maxCount)