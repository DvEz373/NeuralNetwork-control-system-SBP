import matplotlib.pyplot as plt
import pandas as pd

# Creating the data extracted from your dataset
data = {
    "Feel Depressed": [3, 5, 2, 5, 3, 4, 3, 3, 3, 2, 4, 3],
    "Feel Worthless": [3, 3, 1, 5, 4, 5, 4, 4, 3, 1, 2, 3],
    "Out of Control in Work": [4, 4, 2, 3, 5, 5, 5, 4, 4, 2, 2, 4],
    "Doubt Competence": [3, 3, 2, 4, 2, 4, 5, 3, 2, 2, 1, 3],
    "Out of Control in Career": [4, 4, 4, 5, 4, 3, 1, 5, 4, 3, 5, 3],
    "Feel Hopeless": [2, 4, 4, 5, 3, 4, 5, 3, 3, 5, 3, 3],
    "Satisfaction with Self": [3, 4, 5, 4, 4, 5, 5, 4, 5, 3, 5, 4],
    "Complete Tasks Successfully": [3, 5, 5, 4, 5, 5, 4, 5, 5, 3, 5, 4],
    "Confident in Success": [2, 4, 4, 4, 5, 5, 5, 4, 5, 4, 4, 4],
    "Generally Succeed": [3, 4, 4, 5, 4, 4, 4, 4, 5, 3, 4, 3],
    "Cope with Problems": [4, 3, 2, 5, 4, 4, 3, 4, 4, 5, 5, 4]
}

# Creating the DataFrame
df = pd.DataFrame(data)

# Plotting the boxplot
plt.figure(figsize=(15, 8))
df.boxplot()
plt.title("Boxplot of Survey Responses from Dataset")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
