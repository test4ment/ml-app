global Models

Models = {
    "SVC": {
        "Role": "Classification"
    },
    "SVR": {
        "Role": "Regression"
    }
}

global Tasks 

Tasks = {""} | {i["Role"] for i in Models.values()}

# def getModelsNames() -> list[str]:
#     return ["SVM"]