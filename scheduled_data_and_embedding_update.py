import os
import timeit

#imports for data processing/storing
import json
import h5py

#imports for modelling
import torch
from sentence_transformers import SentenceTransformer, util

def read_from_json(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    print(f"{len(data)} entries are read from {file_name}")
    return data

def write_to_json(file_name, data):
    with open(file_name, 'w') as f:
        json.dump(data, f)
    print(f"{len(data)} entries are written to {file_name}")

def getUniqResearchInt(data):
    #returns a set of unique research interests
    researchIntUniq = set()
    for i in range(len(data)):
        #check if 'Research Interests' is in faculty dict, else []
        researchIntUniq.update(data[i].get('Research Interests', []))
    return researchIntUniq

def load_model(modelName):
    #device set as per availability
    deviceName = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(deviceName)
    with torch.no_grad():    
        #dl_ir_model = torch.load('model_cpu.pt')
        model = SentenceTransformer(modelName, device=deviceName)
        model.to(device)
        model.eval()
        return model

def getEmbeddings(modelName, sentences):
    model = load_model(modelName)
    deviceName = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {deviceName} to encode...")
    with torch.inference_mode():
        sentEmbedding = model.encode(sentences, device=deviceName)
        return sentEmbedding

def updateEmbeddings(modelName, sentences, embeddingFileName):
    with h5py.File(embeddingFileName, 'a') as hf:
        print(f"No of embeddings before update is {hf['researchIntEmbedding'].shape[0]}")
        hf["researchIntEmbedding"].resize((hf["researchIntEmbedding"].shape[0] + len(sentences)), axis = 0)
        sentEmbedding = getEmbeddings(modelName, sentences)
        hf["researchIntEmbedding"][-len(sentences):] = sentEmbedding
        print(f"No of embeddings after update is {hf['researchIntEmbedding'].shape[0]}")
            
def assert_files_exist(researchIntEmbeddingFileName):
    embeddingFileFound = True
    mappingFileFound = True
    if not os.path.exists(researchIntEmbeddingFileName):
        embeddingFileFound = False
    if not os.path.exists("indices_to_researchInterest.json"):
        mappingFileFound = False
    if not os.path.exists("researchInterest_to_indices.json"):
        mappingFileFound = False
    if not mappingFileFound:
        print(f"[ERROR] The in-house mapping files are not found. Please run one_time_data_processing.py file first...")
    if not embeddingFileFound:
        print(f"[ERROR] The embedding file {researchIntEmbeddingFileName} is not found. Please run one_time_embedding_creation.py file first...")
    if mappingFileFound and embeddingFileFound:
        return True
    else:
        return False
            
def update_mapping_data(data, modelName, embeddingFileName):
    #check if all files needed are available
    if not assert_files_exist(embeddingFileName):
        return
    
    #load the existing data
    #update research interest mapping data
    #index -> research interest mapping
    idToResearchIntDict = read_from_json("indices_to_researchInterest.json")
        
    #research interest -> index mapping
    researchIntToIdDict = read_from_json("researchInterest_to_indices.json")
    
    #track the added and removed research interest since the last data fetch
    existingReserachInt = set(researchIntToIdDict.keys())
    updatedReserachInt = getUniqResearchInt(data)
    reserachIntAdded = updatedReserachInt.difference(existingReserachInt)
    reserachIntRemoved = existingReserachInt.difference(updatedReserachInt)
    print(f"{len(reserachIntAdded)} research interest(s) are new to our in-house data from this data fetch")
    #TO DO - we accumulate past data in the existing way. To effectivily utilize space do:
    #step 1 - Track the indices of removed research interests and use it for the newly seen research interests
    #step 2 - This also means we need to do similar activity for the embedding matrix -> where we replace the embedding of the removed research interests by that of the new research interests
    
    if len(reserachIntAdded) != 0:
        #update embedding file
        reserachIntAddedList = list(reserachIntAdded)
        execution_time = timeit.timeit(lambda: updateEmbeddings(modelName, reserachIntAddedList, embeddingFileName), number=1)
        print(f"Time taken to encode {len(reserachIntAddedList)} research interests is {execution_time} seconds i.e {execution_time/len(reserachIntAddedList)} seconds per item")
        
        #update our in-house data with the recent fetch
        largestReserachIntIndex = max(map(int, idToResearchIntDict.keys()))
        newResearchIntToIdDict = {x: idx+largestReserachIntIndex+1 for idx, x in enumerate(reserachIntAddedList)}
        researchIntToIdDict.update(newResearchIntToIdDict)
        write_to_json("researchInterest_to_indices.json", researchIntToIdDict)
        newIdToResearchIntDict = {idx+largestReserachIntIndex+1 : x for idx, x in enumerate(reserachIntAddedList)}
        idToResearchIntDict.update(newIdToResearchIntDict)
        write_to_json("indices_to_researchInterest.json", idToResearchIntDict)

file_name = "faculty_data.json"
#pre-trained model
modelName = "sentence-transformers/all-mpnet-base-v2"
#name of embedding file
researchIntEmbeddingFileName = "embeddings_"+"_".join(modelName.split('/'))+".h5"

faculty_data = read_from_json(file_name)
update_mapping_data(faculty_data, modelName, researchIntEmbeddingFileName)

