import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        # Age in decades
        self.dataset['age_in_decades'] = 0.1 * self.dataset['age']
        self.dataset['age_in_decades'] = self.dataset['age_in_decades'].astype(int)

        # BMI NANs
        self.dataset.loc[self.dataset['bmi'].isna(), 'bmi'] = self.dataset['bmi'].median()

        # BMI
        self.dataset['bmi'] = pd.cut(
            self.dataset['bmi'],
            bins=[-float('inf'), 18.5, 25, 30, float('inf')],
            labels=[0, 1, 2, 3],
            right=True
        ).astype(int)

        # Glucose level
        self.dataset['avg_glucose_level'] = pd.qcut(self.dataset['avg_glucose_level'], 8)

        # heart_disease_total
        self.dataset['heart_disease_total'] = (self.dataset['hypertension'] +
                                               self.dataset['heart_disease'])

        # Dropping non essential columns
        drop_elements = ['id', 'age', 'Residence_type', 'hypertension', 'heart_disease', 'gender']
        self.dataset = self.dataset.drop(drop_elements, axis=1)

        # Encoding
        le = LabelEncoder()

        le.fit(self.dataset['ever_married'])
        self.dataset['ever_married'] = le.transform(self.dataset['ever_married'])

        le.fit(self.dataset['work_type'])
        self.dataset['work_type'] = le.transform(self.dataset['work_type'])

        le.fit(self.dataset['smoking_status'])
        self.dataset['smoking_status'] = le.transform(self.dataset['smoking_status'])

        le.fit(self.dataset['avg_glucose_level'])
        self.dataset['avg_glucose_level'] = le.transform(self.dataset['avg_glucose_level'])

        return self.dataset
