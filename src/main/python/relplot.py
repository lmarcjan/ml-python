import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", data=tips)

plt.show()
