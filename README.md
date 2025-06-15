

## What is the project about

Losing clients or customers is referred to as customer attrition, customer churn, customer turnover, or customer defection.

Because keeping an existing customer costs much less than acquiring a new one, companies that provide telephone services, Internet services, pay TV, insurance, and alarm monitoring services frequently use customer attrition analysis and customer attrition rates as one of their key business metrics. Because reclaimed long-term consumers can be worth far more to a firm than newly recruited clients, companies from these sectors frequently have customer service units that work to win back departing customers.

And we tried to make our models more explainable using explainable AI techniques such as Partial Dependency plots and Shapley values.

By determining a customer's propensity for risk of churn, churn prediction models used in predictive analytics can forecast customer churn. These models are useful for concentrating customer retention marketing activities on the segment of the customer base that is most susceptible to churn because they produce a short prioritized list of possible defectors.

We wanted to conduct a customer churn analysis for this project and create a model that can forecast client attrition. Additionally, we have created a dash app that can be used to determine a customer's estimated lifetime value and the reasons why they would cancel a subscription.

## How to run the code

Step 1: Clone the repository

   (https://github.com/Anish-coder09/Churn-Prediction.git)

Step 2: Select the directory

   (https://github.com/Anish-coder09/Churn-Prediction)

Step 3: Install the required libraries

   pandas ,numpy,sklearn,matplotlib,dash 



Step 5: Open the link in the browser

(https://9ea0-35-231-45-36.ngrok-free.app/)

## Final Customer Churn Prediction Flask app

![image](https://github.com/user-attachments/assets/53145989-cae7-40a1-bf7f-4852d69b06c9)


## Explainablity of the model

We have compared the explainability of Random Forest Classifier and a deep learning neural network MLP Classifier. We have used SHAP values to explain the predictions of the model. SHAP values interpret the impact of having a certain value for a given feature in comparison to the prediction we'd make if that feature took some baseline value.

We have explained and understood the Random forest model and the MultiLayerPerceptron model using explainable AI modules such as Permutation Importance, Partial Dependence plots and Shap values.

We see that the neural net model is way more complex than the random forest model and hence it is difficult to explain the neural net model. The random forest model is much simpler and hence it is easier to explain the random forest model.
This is due to the fact that the neural net model has a lot of hidden layers and a lot of neurons in each layer. This makes the model very complex and hence it is difficult to explain the model.

## Feature Importance Plots

### MultiLayerPerceptron

![image](https://github.com/user-attachments/assets/37bb7739-b747-4f85-9b39-d1f34818fa75)
![image](https://github.com/user-attachments/assets/8f85e9ca-a186-4526-b35a-4137b2beb9e5)


### Random Forest Classifier

![image](https://github.com/user-attachments/assets/73ae7e5c-0c4c-4826-aad6-f3bcb959f2c8)

## Partial Dependency

Partial dependence plot is used to see how churning probability changes across the range of particular feature.
We have used partial dependency plots to understand the relationship between the target and the features. We have used the `pdpbox` library to plot the partial dependency plots.

## Shapley Values

Shap values (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. In below plot we can see that why a particual customer's churning probability is less than baseline value and which features are causing them.

We have used shapley values to explain the predictions of the model. SHAP values interpret the impact of having a certain value for a given feature in comparison to the prediction we'd make if that feature took some baseline value. We have used the `shap` library to plot the shapley values.



## Gauge Charts

### MultiLayerPerceptron

![image](https://github.com/user-attachments/assets/e51b6796-6578-46de-b011-584b2cc3b7d6)




## Conclusion

We have created a customer churn prediction model using Random Forest Classifier and a deep learning neural network MLP Classifier. We have used Partial Dependency plots and Shapley values to explain the predictions of the model. We have also created a Flask app that can be used to determine a customer's estimated lifetime value and the reasons why they would cancel a subscription.

The final app shows churning probability, gauge chart of how severe a customer is and shap values based on customer's data. The final app layout can be seen above.
