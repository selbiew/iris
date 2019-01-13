import pickle
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def main():
    label_column = 'class'
    raw_data = pd.read_csv(r'./data/iris.csv')
    processed_data = raw_data

    mus, covariance = gaussian_discriminant_analysis(processed_data, label_column)
    processed_points = processed_data.drop([label_column], axis=1)
    processed_labels = np.ravel(processed_data[[label_column]].values)
    
    model = LinearDiscriminantAnalysis()
    model.fit(processed_points.values, processed_labels)

    results = pd.DataFrame(processed_labels)
    results['mp'] = processed_points.apply(lambda p: predict(p, mus[0], mus[1], covariance), axis=1)
    results['sp'] = processed_points.apply(lambda p: model.predict(p.values.reshape(1, -1)).item(0), axis=1)
    
    mp_accuracy, sp_accuracy = sum(np.where(results['mp'] == results[0], 1, 0)) / len(results), sum(np.where(results['sp'] == results[0], 1, 0))  / len(results)
    print(f'mp_accuracy: {mp_accuracy}, sp_accuracy: {sp_accuracy}')
    
    plt.show()

def gaussian_discriminant_analysis(data, class_column):
    negatives, positives = data[data[class_column] == 0], data[data[class_column] == 1]
    mus = (negatives.drop([class_column], axis=1).mean().values, positives.drop([class_column], axis=1).mean().values)
    covariance = data.drop([class_column], axis=1).cov().values

    return mus, covariance

def predict(point, mu_zero, mu_one, covariance):
    p_failure, p_success = likelihood(point, mu_zero, covariance), likelihood(point, mu_one, covariance)
    
    return 1 if p_success > p_failure else 0

def likelihood(point, mu, covariance):    
    return (1 / (((2 * math.pi) ** (len(mu) / 2.0)) * np.linalg.det(covariance) ** 0.5)) * (math.exp(-0.5 * (point - mu).T @ np.linalg.inv(covariance) @ (point - mu)))

if __name__ == "__main__":
    main()