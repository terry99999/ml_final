import json as json
from modules import *
import yaml

n = 10000
desc_df = pd.DataFrame(np.full([n,1], ""))
#print(desc_df)

for i in range(0,n):
    with open("descriptions_train/" + str(i)+ ".txt") as file:
        lines = file.read()
        desc_df.iat[i, 0] = lines

desc_dict = tokenize_clean(desc_df)

with open("desc_dict.json", "w") as file:
    json.dump(desc_dict, file)

desc_dict_small = {}
for i in range(0,200):
    desc_dict_small[i] = desc_dict[i]

with open("desc_dict_small.yaml", "w") as file:
    yaml.dump(desc_dict_small, file, default_flow_style=False)