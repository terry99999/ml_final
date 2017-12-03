import json as json
from modules import *
import yaml

test_train = "train"
n = 10000
desc_df = pd.DataFrame(np.full([n,1], ""))


for i in range(0,n):
    with open("descriptions_"+ test_train + "/" + str(i)+ ".txt") as file:
        lines = file.read()
        desc_df.iat[i, 0] = lines

desc_dict = tokenize_clean(desc_df)

with open("desc_dict_" + test_train + ".yaml", "w") as file:
    yaml.dump(desc_dict, file)

desc_dict_small = {}
for i in range(0,200):
    desc_dict_small[i] = desc_dict[i]

with open("desc_dict_small_" + test_train + ".yaml", "w") as file:
    yaml.dump(desc_dict_small, file, default_flow_style=False)