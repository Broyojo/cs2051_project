import csv

keys = [
    "OpenPorchSF",
    "GrLivArea",
    "MoSold",
    "BedroomAbvGr",
    "OverallQual",
    "GarageYrBlt",
    "TotalBsmtSF",
    "MasVnrArea",
    "YearBuilt",
    "MSSubClass",
    "KitchenAbvGr",
    "BsmtHalfBath",
    "LotFrontage",
    "LotArea",
    "BsmtFinSF2",
    "SalePrice",
    "GarageArea",
    "HalfBath",
    "GarageCars",
    "1stFlrSF",
    "YearRemodAdd",
    "YrSold",
    "Id",
    "WoodDeckSF",
    "BsmtFinSF1",
    "LowQualFinSF",
    "FullBath",
    "2ndFlrSF",
    "MiscVal",
    "3SsnPorch",
    "PoolArea",
    "OverallCond",
    "BsmtFullBath",
    "TotRmsAbvGrd",
    "ScreenPorch",
    "Fireplaces",
    "EnclosedPorch",
    "BsmtUnfSF",
]

dataset = []

with open("./house-prices-advanced-regression-techniques/train.csv") as file:
    reader = csv.DictReader(file)
    num_keys = set()
    for row in reader:
        data = []
        for key in keys:
            data.append(float(row[key]))
        print(data)
