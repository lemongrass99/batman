# HANDELING MISSING VALUES ##################
# 1. using null and notnull
# import numpy as np
# import pandas as pd

# dict={'First Score ': [100,90,np.nan,95],
#       'Second Score':[30,45,56,np.nan],
#       'Third Score':[np.nan,40,80,98]}
# df=pd.DataFrame(dict)
# # df.isnull()
# # df.notnull()


# INTERPOLATE THE MISSING VALUES
# import pandas as pd 
# import numpy as np
# df=pd.DataFrame({'A': [100,90,None,95],
#       'B':[30,45,56,None],
#       'C':[None,40,80,98],
#       'D':[50,56,None,99]})
# df
# df.interpolate(method='linear',limit_direction='forward')

# USING DROP NA FUNCTION
# import numpy as np
# import pandas as pd

# dict={'First Score ': [100,90,np.nan,95],
#       'Second Score':[30,45,56,np.nan],
#       'Third Score':[np.nan,40,80,98]}
# df=pd.DataFrame(dict)
# # df.dropna()
# # df.dropna(how='all')
# df.dropna(axis=1)