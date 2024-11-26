# USING FILLNA,FILL,BFILL()
# import numpy as np
# import pandas as pd

# dict={'First Score ': [100,90,np.nan,95],
#       'Second Score':[30,45,56,np.nan],
#       'Third Score':[np.nan,40,80,98]}
# df=pd.DataFrame(dict)
# df.fillna(method='pad')
# df.bfill
# df.fillna

# VIEW FIRST FIVE ROWS OF THE DATAFRAME#####

# import pandas as pd
# data=pd.read_csv("C:\\Users\\vagis\\OneDrive\\Desktop\\vagish\\Datasets\\employees.csv")
# data.head()

# COMPUTE AND PRINT STD AND MEAN FOR EACH MEASURMENTS
# import pandas as pd
# import matplotlib.pyplot as ply
# from pandas.plotting import parallel_coordinates
# data=pd.read_csv("C:\\Users\\vagis\\OneDrive\\Desktop\\vagish\\Datasets\\employees.csv")
# data.head()
# print("STD:\n",data.std(numeric_only=True),"\n")
# print("Mean:\n",data.mean(numeric_only=True),"\n")
# print("MIN:\n",data.min(numeric_only=True),"n/")
# print("MAX:\n",data.max(numeric_only=True),"\n")


# SCATTERPLOT
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
data = pd.read_csv("C:\\Users\\vagis\\OneDrive\\Desktop\\vagish\\Datasets\\Iris.csv")
print(data)
plt.scatter(data["SepalLengthCm"],data["SepalWidthCm"])
plt.xlabel("Sepal length in (cm)")
plt.ylabel("Sepal width in (cm)")
plt.show()

parallel_coordinates(data,"Species")
plt.show()



# DISPLAY ALL FUNCTIONS FROM THE  ABOVE DATA 
data.describe(include='all')