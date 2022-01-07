import pickle
import csv
from typing_extensions import final
vectorizer = pickle.load(open('data/vectorizer.pickle', 'rb'))
model = pickle.load(open('data/model.sav','rb'))



data = []
with open('data/processed_tweets.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:        
        data.append(row[0])   


vector_data = vectorizer.transform(data)
preds =  model.predict(vector_data)


final_prediction = (round(sum(preds)/len(preds)))

if final_prediction ==1:
    print('Team won')
else:
    print('Team lost')