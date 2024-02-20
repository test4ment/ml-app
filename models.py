global Models

Models = {
    "SVC": {
        "Role": "Classifier"
    },
    "SVR": {
        "Role": "Regressor"
    }
}

global Tasks 

Tasks = {""} | {i["Role"] for i in Models.values()}

# def getModelsNames() -> list[str]:
#     return ["SVM"]