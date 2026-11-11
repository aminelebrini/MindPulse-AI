import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
class DataExploration:

    def __init__(self, data):
        self.data = pd.DataFrame(data)

    def explore_data(self):
        print("Data Overview:")
        print(self.data.head())
        print("\nData Info:")
        print(self.data.info())
        print("\nData Shape:")
        print(self.data.shape)
        print("\nData Columns:")
        print(self.data.columns.tolist())
        print("\nData Types:")
        print(self.data.dtypes)
        print("\nduplicate Rows:")
        print(self.data.duplicated().sum())
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        print("\nDescriptive Statistics:")
        print(self.data.describe())

        plt.figure(figsize=(7, 4))
        sns.countplot(
            data=self.data,
            x='Mental_Health_Risk',
            order=['Low', 'Medium', 'High'],
            palette='Set2'
        )

        plt.title('Distribution of Mental Health Risk (Target)', fontsize=13)
        plt.xlabel('Risk Level', fontsize=11)
        plt.ylabel('Count', fontsize=11)
        sns.despine()
        plt.show()


        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=self.data,
            x='Mental_Health_Risk',
            y='Sleep_Hours',
            order=['Low', 'Medium', 'High'],
            palette='Set2',
        )
        plt.title(
            'Distribution of Sleep Hours by Mental Health Risk Level',
            fontsize=13,
            fontweight='bold',
        )
        plt.xlabel('Mental Health Risk', fontsize=11)
        plt.ylabel('Sleep Hours per Night', fontsize=11)
        sns.despine()
        plt.tight_layout()
        plt.show()


        diet_cross = pd.crosstab(
            self.data['Diet_Quality'], 
            self.data['Mental_Health_Risk'], 
            normalize='index'
        )*100

        diet_cross[[
            'Low', 'Medium', 'High'
        ]].plot(
            kind='bar', 
            stacked=True, 
            color=['#66c2a5', '#fc8d62', '#8da0cb'], 
            figsize=(8, 5)
        )

        plt.title(
            'Proportion of Mental Health Risk Levels by Diet Quality',
            fontsize=13,
            fontweight='bold',
        )
        plt.xlabel('Diet Quality', fontsize=11)
        plt.ylabel('Percentage (%)', fontsize=11)
        plt.legend(title='Risk Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        sns.despine()
        plt.tight_layout()
        plt.show()


        df_numeric = self.data.select_dtypes(include=['number'])
        corr_matrix = df_numeric.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
        )
        plt.title('Correlation Matrix', fontsize=13, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        df_numeric = df_numeric.columns

        plt.figure(figsize=(10, 5))

        sns.boxplot(
            data=self.data[df_numeric],
            orient='v',
            palette='Set3',
        )
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('Boxplot of Numeric Features for Analysis outliers', fontsize=13, fontweight='bold')
        plt.xlabel('Value')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()


        data_frame = self.data.copy()
        risk_mapping = {
            'Low': 0,
            'Medium': 1,
            'High': 2
        }
        data_frame['Risk_numeric'] = data_frame['Mental_Health_Risk'].map(risk_mapping)

        target_coor = (
            data_frame.select_dtypes(include=['number'])
            .corr()['Risk_numeric'].drop('Risk_numeric')
            .sort_values(ascending=False)
        )

        print(target_coor)

        plt.figure(figsize=(10, 6))
        target_coor.plot(kind='bar', color='skyblue',
                         edgecolor='black'
                         )
        plt.title(
            'Variables Most Strongly Correlated with Mental Health Risk',
            fontsize=13,
            fontweight='bold',
        )
        plt.ylabel('Correlation Coefficient', fontsize=11)
        plt.xlabel('Features', fontsize=11)
        sns.despine()
        plt.tight_layout()
        plt.show()


    