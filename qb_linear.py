"""
    CSC 370: Quarterback ratings, linear models, information from data
    Group members: Tom Nguyen, Trung Pham
"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


def calculate_passer_rating(row):  # Part 1
    # Extract the relevant statistics from the row
    pass_attempt = row['Att']
    pass_completed = row['Cmp']
    passing_yard = row['Yds']
    passing_touchdown = row['TD']
    interception = row['Int']

    a = (pass_completed/pass_attempt - .3) * 5
    b = (passing_yard/pass_attempt - 3) * .25
    c = (passing_touchdown/pass_attempt) * 20
    d = 2.375 - (interception/pass_attempt * 25)

    passer_rating = (a+b+c+d)/6 * 100

    return passer_rating


def extract_significant_features(fitted_model):  # Part 3
    # Get the analysis table from the model summary
    summary_table = fitted_model.summary2().tables[1]

    # Extract the feature names and p-values
    feature_data = summary_table[['Coef.', 'P>|t|']]

    # Filter features with p-values less than 0.05 (significant features)
    significant_features = feature_data[feature_data['P>|t|'] < 0.05]

    # Extract the names of significant features as a list
    significant_feature_names = significant_features.index.tolist()

    return significant_feature_names


def main():
    # Load the DataFrame from the CSV file
    df = pd.read_csv('qb2018_simple.csv')
    features = ['Age', 'G', 'GS', 'Cmp', 'Att', 'Cmp%', 'Yds',
                'TD', 'TD%', 'Int', 'Int%', 'Lng', 'Y/A', 'AY/A',
                'Y/C', 'Y/G', 'Sk', 'Yds.1', 'NY/A', 'ANY/A', 'Sk%']

    # Part 1 start
    # Apply the calculate_passer_rating function to each row
    df['Calculated_Rate'] = df.apply(calculate_passer_rating, axis=1)

    for index, row in df.iterrows():
        player_name = row['Player']
        listed_rate = row['Rate']
        calculated_rate = row['Calculated_Rate']
        print(
            f"{player_name}, Listed Rate: {listed_rate}, Calculated Rate: {calculated_rate}")
    # Part 1 end

    # # Part 2 start
    # # Create the feature matrix (X) and target vector (Y)
    # X2 = df[features]
    # Y2 = df['Rate']

    # X2 = sm.add_constant(X2)

    # model2 = sm.OLS(Y2, X2).fit()

    # print(model2.summary())

    # # Extract p-values for each feature
    # p_values2 = model2.pvalues

    # # Identify significant features (p-value < 0.05)
    # significant_features2 = [feature for feature,
    #                          p_value2 in p_values2.items() if p_value2 < 0.05]

    # print("Significant Features:")
    # for feature in significant_features2:
    #     print(feature)

    # """
    # Based on the provided output of the OLS regression results, the features that had a significant contribution to predicting the 'Rate' (passer rating) based on a regression p-value less than 0.05 are as follows:

    # 1. Cmp%: Completion Percentage - p-value < 0.001
    # 2. TD%: Touchdown Percentage - p-value < 0.001
    # 3. Int%: Interception Percentage - p-value = 0.002
    # 4. AY/A: Adjusted Yards per Attempt - p-value < 0.001
    # 5. NY/A: Net Yards per Attempt - p-value = 0.013
    # 6. ANY/A: Adjusted Net Yards per Attempt - p-value = 0.004

    # These features have p-values less than the significance level of 0.05, indicating that they are statistically significant contributors to predicting the passer rating ('Rate'). These variables are considered important factors in explaining variations in the passer rating among quarterbacks.
    # The significant features identified in the model have strong correlations with the parameters used to calculate the passer rating. They capture key elements such as completion percentage, touchdown percentage, interception percentage, and various aspects of yards per attempt, all of which are integral to both the traditional passer rating and the linear regression model for predicting 'Rate.' However, the linear regression model provides a more flexible and data-driven approach to assess the importance of these factors and their specific contributions to the passer rating.

    # """
    # # Part 2 end

    # # Part 3 start
    # X3 = df[features]
    # Y3 = df['Rate']

    # X3 = sm.add_constant(X3)

    # fitted_model3 = sm.OLS(Y3, X3).fit()
    # significant_features3 = extract_significant_features(fitted_model3)
    # print(significant_features3)
    # # Part 3 end

    # # Part 4 start
    # X4 = df[features]
    # Y4 = df['QBR']

    # model4 = sm.OLS(Y4, X4).fit()

    # print(model4.summary())

    # # Extract p-values for each feature
    # p_values4 = model4.pvalues

    # # Identify significant features (p-value < 0.05)
    # significant_features4 = [feature for feature,
    #                          p_value in p_values4.items() if p_value < 0.05]

    # print("Significant Features for Total QBR:")
    # for feature in significant_features4:
    #     print(feature)

    # """
    # Based on the provided output of the OLS regression results for predicting Total QBR, the significant contributor to the Total QBR is:

    # Y/C (Yards per Completion): This feature represents the average yards gained per completion, and it has a p-value of 0.025, indicating that it is statistically significant in predicting Total QBR.

    # For Total QBR: the significant feature is 'Y/C' (Yards per Completion).
    # For Passer Rating ('Rate') in Task 2: the significant features included 'Cmp%' (Completion Percentage), 'TD%' (Touchdown Percentage), 'Int%' (Interception Percentage), 'AY/A' (Adjusted Yards per Attempt), 'NY/A' (Net Yards per Attempt), and 'ANY/A' (Adjusted Net Yards per Attempt).

    # Comparison:

    # The significant features for Total QBR and passer rating have some differences.
    # 'Cmp%' (Completion Percentage) and 'AY/A' (Adjusted Yards per Attempt) were significant for passer rating ('Rate') but not for Total QBR.
    # 'Y/C' (Yards per Completion) was significant for Total QBR but not for passer rating ('Rate').
    # 'TD%' (Touchdown Percentage), 'Int%' (Interception Percentage), 'NY/A' (Net Yards per Attempt), and 'ANY/A' (Adjusted Net Yards per Attempt) were significant for both Total QBR and passer rating ('Rate').
    # These differences suggest that the proprietary Total QBR and the traditional passer rating may consider different factors or weigh them differently in their calculations. 'Y/C' may play a more prominent role in the Total QBR calculation compared to the traditional passer rating formula.
    # """
    # # Part 4 end

    # # Part 5 start
    # # Define the independent variables (significant features) and the target variable
    # significant_features5 = ['Cmp%', 'Y/A',
    #                          'AY/A', 'Y/G', 'NY/A', 'ANY/A', 'Sk%']

    # X5 = df[significant_features5]
    # Y5 = df['QBR']

    # X5 = sm.add_constant(X5)

    # model5 = sm.OLS(Y5, X5).fit()

    # print(model5.summary())

    # # Make predictions
    # y_pred5 = model5.predict(X5)

    # # Compare predictions with actual QBR values
    # comparison_df = pd.DataFrame({'Actual QBR': Y5, 'Predicted QBR': y_pred5})
    # print(comparison_df)

    # # Comment on model performance
    # """
    # The new QBR prediction model explains around 86.3% of the QBR variance, indicating a reasonably good fit. Significant features like 'Y/G' and 'Y/C' positively impact QBR, while 'Y/A' negatively affects it. Other features like 'Age,' 'G,' and 'Cmp%' are not significant. The model's predictions closely match actual QBR for some quarterbacks but vary for others. This suggests room for improvement and highlights differences in contributing factors between QBR and passer rating models.

    # """
    # # Part 5 end

    # # Part 6 start
    # df['QBR_Passer_Rating_Diff'] = df['QBR'] - df['Rate']
    # X6 = df[features]
    # Y6 = df['QBR_Passer_Rating_Diff']

    # X6 = sm.add_constant(X6)

    # model6 = sm.OLS(Y6, X6).fit()

    # # Extract significant contributors to the difference
    # significant_features_diff6 = extract_significant_features(model6)

    # # Identify the top 5 quarterbacks with the greatest differences
    # top_5_diff_qbs = df.nlargest(5, 'QBR_Passer_Rating_Diff')

    # print("Top 5 Quarterbacks with Greatest QBR-Passer Rating Difference:")
    # print(top_5_diff_qbs[['Player', 'QBR', 'Rate']])

    # # Comment on observations
    # """
    # The top 5 quarterbacks with the greatest difference between QBR and passer rating have substantial variations
    # in their performance metrics. While specific features contributing to these differences may vary, factors such
    # as completion percentage, yards per attempt, touchdown percentage, and interception percentage are likely to play
    # a role.

    # The differences highlight that QBR considers additional factors beyond traditional passer rating, including
    # situational and performance context. These differences can be explained by a combination of factors such as
    # rushing statistics, sack percentage, and how a quarterback performs in various game situations.

    # Further analysis would involve a detailed examination of these quarterbacks' stats to understand the specific
    # features that contribute to the QBR-Passer Rating difference for each of them.
    # """
    # # Part 6 end


main()
