import os
from eschernode.settings import mongoClient, MONGO_DB
from bson.objectid import ObjectId

db = mongoClient[MONGO_DB]

def getUserById(id):
    return db.users.find_one({'_id': ObjectId(id)})
    # raise NotImplementedError

def getNNModelById(id):
    return db.nnmodels.find_one({'_id': ObjectId(id)})
    # raise NotImplementedError

def getDatasetById(id):
    return db.datasets.find_one({'_id': ObjectId(id)})

def getExperimentById(id):
    # print(db)
    return db.experiments.find_one({'_id': ObjectId(id)})
    # raise NotImplementedError

# def getDatasetById(id):
#     return db.datasets.find_one({'_id': ObjectId(id)})