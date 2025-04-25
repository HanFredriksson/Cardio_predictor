import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


class Cardio_data:

    def __init__(self, data, bmi_max_min, pressure_values):

        """
        data - A dataframe with data for cardiovascular diseases
        bmi_max_min - list, A max value and a min value
        blood_pressure - dict, with a list of ap_hi max/min and ap_lo max/min values
        encode_data - Stores the encoded data
        """

        self.data = data.copy() # Avoid modifying original data
        self.encoded_data = []
        self.bmi = bmi_max_min
        self.blood_pressure = pressure_values

        self.bmi_calculator_labels()
        self.blood_pressure_labels()



    def bmi_calculator_labels(self):
        """
        Bmi calculator
        
        weight - int, in kilograms
        height - int, in centimeters
        bmi_max_min - list, A max value and a min value

        Lebals are BMI: 18-25 normal range, 25-30 over-weight, 31-35 obese (class I), 36-40 obese (class II), 40 above obese (class III)  

        Retrun int - bmi, kg/m^2
        """

        self.data["BMI"] = self.data["weight"] / (self.data["height"] / 100) ** 2
        self.data = self.data[(self.data["BMI"] <= self.bmi[0]) & (self.data["BMI"] >= self.bmi[1])]
        self.data["BMI cat"] = pd.cut(self.data["BMI"], 
                               bins=[18, 25, 30, 35, 40, float("inf")], 
                               labels= ["normal range", "over-weight", "obese (class I)", "obese (class II)", "obese (class III)"])


    def blood_pressure_labels(self):
        """
        Gives you the lables for an range of blood pressure and removes the outliers from data.
        Given the range for the max values and min values of ap_hi and ap_lo

        Labels ap_hi/ap_lo: below 120/80 Healty, 120-129/80, Elvated, 130-139/80-89 Stage 1 hypertension
        above 140/90, Stage 2 hypertension.
        """

        self.data = self.data[(self.data["ap_hi"] <= 160) & (self.data["ap_hi"] >= 90)]
        self.data = self.data[(self.data["ap_lo"] <= 120) & (self.data["ap_lo"] >= 70)]
        self.data["BPR"] = None
        for i in self.data.index:
            ap_hi, ap_lo = self.data.at[i, "ap_hi"], self.data.at[i, "ap_lo"]
        
            if ap_hi < 120 and ap_lo < 80:
                self.data.at[i, "BPR"] = "Healty"
            
            elif 120 <= ap_hi <= 129 and ap_lo < 80:
                self.data.at[i, "BPR"] = "Elevated"
            
            elif (130 <= ap_hi <= 139) or (80 <= ap_lo <= 89):
                self.data.at[i, "BPR"] = "Stage 1 hypertension"
        
            elif ap_hi >= 140 or ap_lo >= 90:
                self.data.at[i, "BPR"] = "Stage 2 hypertension"





    def encode_df(self):
        """
        Encodes categorical columns into one-hot encoded columns. Can handle multiple encodings
        to make different datasets.

        categorical_columns - List of lists with column names to be one-hot encoded.
        """
        drop_columns = [["ap_hi", "ap_lo", "height", "weight", "BMI"], ["height", "weight", "BMI cat", "BPR"]]
        categorical_columns = [["BMI cat", "BPR", "gender"], ["gender"]]     

        self.encoded_data = []

        for columns, drop in zip(categorical_columns, drop_columns):
            data = self.data.drop(drop, axis=1)
            encoder = OneHotEncoder(sparse_output=False)
            one_hot_encoded = encoder.fit_transform(data[columns])
            one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(columns), index=self.data.index)
            data = data.drop(columns, axis=1)
            encoded_data = pd.concat([data, one_hot_df], axis=1)
            self.encoded_data.append(encoded_data)
        
        return self.encoded_data

    

    def eda_plot(self):
        """
        EDA plots asked from the assignment
        """
        fig, axes = plt.subplots(3, 3, dpi=100, figsize=(15, 15))

        sns.countplot(data=self.data, x="cardio", ax=axes[0, 0])
        axes[0, 0].set_xlabel("0: Negative, 1: Positive")
        axes[0, 0].set_ylabel("Number of people")
        axes[0, 0].set_title("Number of people with/without cardiovascular disease")

        axes[0, 1].pie(
            self.data["cholesterol"].value_counts(), labels=["1", "2", "3"], autopct="%.0f%%"
        )
        axes[0, 1].set_title("The proportion of cholesterol levels")
        axes[0, 1].legend(["1: Normal", "2: Above normal", "3: Well above normal"])

        sns.histplot(self.data["age"], ax=axes[0, 2], bins=20, color="blue", alpha=0.7, edgecolor="black")
        axes[0, 2].set_xlabel("Age in days")
        axes[0, 2].set_ylabel("Number of Positive")
        axes[0, 2].set_title("Ages with cardiovascular disease (Age in days)")

        sns.countplot(data=self.data, x="smoke", ax=axes[1, 0])
        axes[1, 0].set_xlabel("0: Non-smoker, 1: Smoker")
        axes[1, 0].set_ylabel("Number of people")
        axes[1, 0].set_title("Number of smokers and non-smokers")

        sns.histplot(self.data["weight"], kde=True, ax=axes[1, 1])
        axes[1, 1].set_xlabel("Weight")
        axes[1, 1].set_ylabel("Number of people")
        axes[1, 1].set_title("Weight distribution")

        sns.histplot(self.data["height"], kde=True, ax=axes[1, 2])
        axes[1, 2].set_xlabel("Height")
        axes[1, 2].set_ylabel("Number of people")
        axes[1, 2].set_title("Height distribution")

        sns.countplot(data=self.data[self.data["cardio"] == 1], x="gender", ax=axes[2, 0])
        axes[2, 0].set_xlabel("1 - Women, 2 - Men")
        axes[2, 0].set_ylabel("Number of Positive Cases")
        axes[2, 0].set_title("Cardiovascular disease cases by gender")

        plt.tight_layout()

        
        fig.delaxes(axes[2, 1])
        fig.delaxes(axes[2, 2])

        return plt.show()
    

    def plot_number_diseases(self):
        """
        Looking at cardiovascular possetiv and against some other features
        """
        cardio_positive = self.data[self.data["cardio"] == 1]
        features = ["BMI cat", "BPR", "active", "alco", "gender", "smoke"]
        labels = ["BMI category", "Blood Pressure Range", "Physical activity: Yes = 1, No = 0", 
                "Alcohol drinker: Yes = 1, No = 0", "Gender: 1 - Women, 2 - Men", "Smoker: Yes = 1, No = 0"] 

        fig, axes = plt.subplots(2, 3, figsize=(20, 8), dpi=100)


        for i, ax in enumerate(axes.flatten()):
            sns.countplot(cardio_positive, x=features[i], ax=ax)
            ax.set(xlabel=labels[i], ylabel="Postive cardio disease count")
        plt.tight_layout()

        return plt.show()
    
    def plot_heatmap(self):
        """
        Heatmap
        """
        plt.figure(figsize=(15,8))
        sns.heatmap(self.data.corr(numeric_only=True), vmin=-1, vmax=1, annot=True)