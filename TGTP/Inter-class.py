#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import shap
import matplotlib.pyplot as plt


# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Separate features and labels
    X = df.drop(columns=['Label']).values
    y = df['Label'].values
    # Handle infinite values
    X[np.isinf(X)] = np.nan
    # Fill NaN values with column means
    X = np.nan_to_num(X, nan=np.nanmean(X))
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)
    # Check for invalid values in the training data
    assert not np.isinf(X_train).any(), "Input contains infinity after processing."
    assert not np.isnan(X_train).any(), "Input contains NaN values after processing."
    assert np.isfinite(X_train).all(), "Input contains values too large for float64 after processing."
    return X_train, X_test, y_train, y_test, scaler, df


# Function to train the model
def train_model(X_train, y_train):
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000)
    model.fit(X_train, y_train)
    return model


# Function to calculate SHAP values
def calculate_shap_values(model, X_train, X_test):
    background = shap.kmeans(X_train, 10)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X_test)
    return shap_values, explainer


# Function to plot SHAP summary plot
def plot_shap_summary(shap_values, X_test, features):
    shap.summary_plot(shap_values, X_test, feature_names=features)


# Function to plot SHAP bar plot
def plot_shap_bar(shap_values):
    shap.plots.bar(shap_values[0].cohorts(2).abs.mean(0))


# Function to plot SHAP heatmap
def plot_shap_heatmap(shap_values, X_test, features, explainer):
    shap_explanation = shap.Explanation(
        values=shap_values[0][0:500, :],
        base_values=explainer.expected_value[0],
        data=X_test[0:500, :],
        feature_names=features
    )
    plt.figure(figsize=(10, 8))
    shap.plots.heatmap(shap_explanation)
    plt.show()


# Function to calculate and print average SHAP values per class
def calculate_and_print_avg_shap(shap_values, features):
    average_shap_values_per_class = []
    for class_idx in range(len(shap_values)):
        avg_shap_per_feature = np.mean(shap_values[class_idx], axis=0)
        average_shap_values_per_class.append(avg_shap_per_feature)
        print(f"Average SHAP values for class {class_idx}:")
        sorted_features = sorted(zip(features, avg_shap_per_feature), key=lambda x: x[1], reverse=True)
        for feature_name, shap_val in sorted_features:
            print(f"Feature '{feature_name}': {shap_val:.4f}")
        print("Bottom 5 features:")
        for feature_name, shap_val in sorted_features[-5:]:
            print(f"Feature '{feature_name}': {shap_val:.4f}")
        print("Top 5 features:")
        for feature_name, shap_val in sorted_features[:5]:
            print(f"Feature '{feature_name}': {shap_val:.4f}")
        print("\n")


# Function to find features with max SHAP values for a target class
def find_max_shap_features(shap_values, X_test, features, scaler, target_class=0):
    max_shap_feature_contributions = {}
    for feature_index, feature_name in enumerate(features):
        max_shap_value = -np.inf
        max_feature_value = None
        max_feature_original_value = None
        for i in range(len(shap_values[target_class])):
            if shap_values[target_class][i, feature_index] > max_shap_value:
                max_shap_value = shap_values[target_class][i, feature_index]
                max_feature_value = X_test[i, feature_index]
                max_feature_original_value = max_feature_value * scaler.scale_[feature_index] + scaler.mean_[
                    feature_index]
        max_shap_feature_contributions[feature_name] = (max_shap_value, max_feature_original_value)
    return max_shap_feature_contributions


# Function to plot SHAP dependence plots for top features
def plot_shap_dependence(shap_values, X_test, features, scaler, target_class=0, top_n=5):
    max_shap_feature_contributions = find_max_shap_features(shap_values, X_test, features, scaler, target_class)
    top_features = sorted(max_shap_feature_contributions.items(), key=lambda x: x[1][0], reverse=True)[:top_n]
    X_test_original = X_test * scaler.scale_.reshape(1, -1) + scaler.mean_.reshape(1, -1)
    for feature_name, (_, max_feature_original_value) in top_features:
        feature_index = features.index(feature_name)
        try:
            shap.dependence_plot(feature_name, shap_values[target_class], X_test_original, interaction_index=None,
                                 feature_names=features)
        except ValueError as e:
            print(e)
            shap.dependence_plot(feature_index, shap_values[target_class], X_test_original, interaction_index=None,
                                 feature_names=features)
    plt.show()


# Function to find features with min absolute SHAP values for a target class
def find_min_abs_shap_features(shap_values, X_test, features, scaler, target_class=0, top_n=75):
    min_abs_shap_feature_contributions = {}
    for feature_index, feature_name in enumerate(features):
        min_abs_shap_value = np.inf
        min_feature_value = None
        min_feature_original_value = None
        for i in range(len(shap_values[target_class])):
            abs_shap_value = np.abs(shap_values[target_class][i, feature_index])
            if abs_shap_value < min_abs_shap_value:
                min_abs_shap_value = abs_shap_value
                min_feature_value = X_test[i, feature_index]
                min_feature_original_value = min_feature_value * scaler.scale_[feature_index] + scaler.mean_[
                    feature_index]
        min_abs_shap_feature_contributions[feature_name] = (min_abs_shap_value, min_feature_original_value)
    min_features = sorted(min_abs_shap_feature_contributions.items(), key=lambda x: x[1][0])[:top_n]
    for feature_name, (min_abs_shap_value, min_feature_original_value) in min_features:
        print(
            f"The feature '{feature_name}' has the smallest absolute SHAP value of {min_abs_shap_value} for class {target_class} with the feature value being {min_feature_original_value}.")


# Function to get top N features with highest SHAP values for a single sample
def get_topN_reason(old_list, features, top_num=3, min_value=0.0):
    feature_importance_dict = {}
    for i, f in zip(old_list, features):
        feature_importance_dict[f] = i
    new_dict = dict(sorted(feature_importance_dict.items(), key=lambda e: e[1], reverse=True))
    return_dict = {}
    for k, v in new_dict.items():
        if top_num > 0:
            if v >= min_value:
                return_dict[k] = v
                top_num -= 1
            else:
                break
        else:
            break
    return return_dict


if __name__ == "__main__":
    file_path = 'train_merge.csv'
    X_train, X_test, y_train, y_test, scaler, df = load_and_preprocess_data(file_path)
    model = train_model(X_train, y_train)
    shap_values, explainer = calculate_shap_values(model, X_train, X_test)
    features = df.drop(columns=['Label']).columns.tolist()

    plot_shap_summary(shap_values, X_test, features)
    plot_shap_bar(shap_values)
    plot_shap_heatmap(shap_values, X_test, features, explainer)
    calculate_and_print_avg_shap(shap_values, features)
    plot_shap_dependence(shap_values, X_test, features, scaler)
    find_min_abs_shap_features(shap_values, X_test, features, scaler)

    print(get_topN_reason(old_list=shap_values[0, :], features=features))