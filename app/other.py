def save_model():
    import pickle
    import json
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from os import getcwd

    from utils.dataloader import DataLoader
    from settings.constants import TRAIN_CSV

    with open('settings/specifications.json') as f:
        specifications = json.load(f)

    raw_train = pd.read_csv(TRAIN_CSV)
    x_columns = specifications['description']['X']
    y_column = specifications['description']['y']

    X_raw = raw_train[x_columns]

    loader = DataLoader()
    loader.fit(X_raw)
    X = loader.load_data()
    y = raw_train.stroke

    model = RandomForestClassifier()
    model.fit(X, y)
    with open(getcwd() + '/models/RandForest.pickle', 'wb')as f:
        pickle.dump(model, f)


def test_api():
    import json
    import requests
    import pandas as pd
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    from utils import DataLoader, Estimator
    from settings.constants import TRAIN_CSV, VAL_CSV

    with open('settings/specifications.json') as f:
        specifications = json.load(f)

    info = specifications['description']
    x_columns, y_column, metrics = info['X'], info['y'], info['metrics']

    train_set = pd.read_csv(TRAIN_CSV, header=0)
    val_set = pd.read_csv(VAL_CSV, header=0)

    train_x, train_y = train_set[x_columns], train_set[y_column]
    val_x, val_y = val_set[x_columns], val_set[y_column]

    loader = DataLoader()
    loader.fit(val_x)
    val_processed = loader.load_data()
    print('data: ', val_processed[:10])

    req_data = {'data': json.dumps(val_x.to_dict())}

    # To test localhost is used
    response = requests.get('http://127.0.0.1:8000/predict', data=req_data)
    api_predict = response.json()['prediction']
    print('predict: ', api_predict[:10])

    api_score = eval(metrics)(val_y, api_predict)
    print('accuracy: ', api_score)


if __name__ == '__main__':
    test_api()
