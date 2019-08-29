import numpy as np
import pandas as pd
import os
import tqdm as tqdm
#
trainingSetPath = 'data/training'
patientList = os.listdir(trainingSetPath)


def preprocessing_data(file):
    data = pd.read_csv(os.path.join(trainingSetPath, file), sep='|')
    data.drop(['EtCO2', 'BaseExcess','HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
               'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct' , 'Resp','Lactate','Magnesium','Phosphate',
               'Potassium', 'Bilirubin_total', 'TroponinI','Hct', 'Hgb','PTT',  'WBC', 'Fibrinogen', 'Platelets',
               "Unit1", 'Unit2', "Glucose", 'DBP', 'SBP'], axis=1, inplace=True)

    data.fillna(method='bfill', inplace=True)
    # data.fillna(method='ffill', inplace=True)

    label = data.SepsisLabel
    data.drop(['SepsisLabel'], axis = 1, inplace = True)

    return data.values, label.values


patient_total = None
label_total = None
i = 0
a = 0
# for patientPath in patientList:
#     patient, patientLabel = code(patientPath)
#     a = np.maximum(a, patientLabel.shape[0])
#     print(a,end=' ')
#     print(patientLabel.shape[0])

for patientPath in patientList:

    print('{} sur {} '.format(i, len(patientList)))
    patient, patientLabel = preprocessing_data(patientPath)

    if patient.shape[0] <= 50:

        patient = np.reshape(patient, (1, patient.shape[0], patient.shape[1]))
        patientLabel = np.reshape(patientLabel, (1, patientLabel.shape[0]))

        pad = np.zeros((1, 50 - patient.shape[1], 8))
        padLabel = np.zeros((1, 50 - patientLabel.shape[1]))

        patient = np.concatenate([patient, pad], axis=1)
        patientLabel = np.concatenate([patientLabel, padLabel], axis=1)

        if i == 0:
            patient_total = patient
            label_total = patientLabel
        else:
            patient_total = np.concatenate([patient_total, patient], axis=0)
            label_total = np.concatenate([label_total, patientLabel], axis=0)

        i = i + 1

patient_total = np.array(patient_total)
label_total = np.array(label_total)

np.savez('data.npz', features=patient_total, label=label_total)
#

data_load = np.load('data.npz', allow_pickle=True)

feat = data_load['features']
label = data_load['label']

print(label.shape)
print(feat.shape)