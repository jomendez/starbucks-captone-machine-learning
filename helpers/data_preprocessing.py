import os
import pandas as pd
import argparse

os.system('pip install joblib')
os.system('pip install imblearn')

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import joblib

targets = ['gender', 'age', 'income', 'day', 'member_from', 'dow',
             'n_transactions', 'avg_transctions',
             'n_offers_completed', 'n_offers_viewed', 'avg_reward',
             'reception_to_view_avg', 'view_to_completion_avg']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--val_split_ratio', type=float, default=0.2)
    parser.add_argument('--test_split_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1123)
    args = parser.parse_args()

    seed = args.seed
    target_feature = args.target
    if target_feature not in ['bogo', 'discount', 'info']:
        raise ValueError(
            ('Target argument must be "bogo", "discount" or "info" - '
             f'{target_feature} passed')
        )

    
    input_path = os.path.join('/opt/ml/processing/input', f'{target_feature}.csv')
    df = pd.read_csv(input_path, header=None, names=[target_feature] + targets)
    
    model = RandomUnderSampler(random_state=seed)
    X_fitted, y_fitted = model.fit_sample(df.drop(target_feature, 1), df[target_feature])

    # Split train, validation, test data
    X, X_val, y, y_val = train_test_split(X_fitted, y_fitted, test_size=args.val_split_ratio, stratify=y_fitted, random_state=seed)
    X, X_test, y, y_test = train_test_split(X, y, test_size=args.test_split_ratio / (1 - args.val_split_ratio), stratify=y, random_state=seed)


    # One Hot Encoding
    gender_pipe = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='constant', fill_value='O')),
            ('ohe', OneHotEncoder())
        ]
    )
    
    # Imputing with median, then standardizing distribution
    numeric_pipe = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )

    # Join the pipelines via a ColumnTransformer
    data_preprocessing = ColumnTransformer(transformers=[
        ('gender_pipeline', gender_pipe, ['gender']),
        ('numeric_pipeline', numeric_pipe, targets[1:])
    ])

    # Fit the transformer
    X = data_preprocessing.fit_transform(X)
    X_val = data_preprocessing.transform(X_val)
    X_test = data_preprocessing.transform(X_test)

    # Save data
    pd.concat([y.reset_index(drop=True), pd.DataFrame(X)], axis=1).to_csv(
        os.path.join(f'/opt/ml/processing/output/{target_feature}_train.csv'),
        header=False, index=False
    )
    pd.concat([y_val.reset_index(drop=True), pd.DataFrame(X_val)], axis=1).to_csv(
        os.path.join(f'/opt/ml/processing/output/{target_feature}_val.csv'),
        header=False, index=False
    )
    pd.Series(y_test.reset_index(drop=True)).to_csv(
        os.path.join(f'/opt/ml/processing/output/{target_feature}_test_target.csv'),
        header=False, index=False
    )
    pd.DataFrame(X_test).to_csv(
        os.path.join(f'/opt/ml/processing/output/{target_feature}_test.csv'),
        header=False, index=False
    )

    # Save transformer
    joblib.dump(
        data_preprocessing, f'/opt/ml/processing/output/{target_feature}_transformer.joblib'
    )