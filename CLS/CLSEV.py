import pandas as pd
df=pd.read_csv("cls.csv", header=None)
df.to_cvs("cls.csv", header=["layer1","layer2","layer3","layer4","layer5","layer6","layer7","layer8","layer9","layer10","layer11","layer12"],index=False)