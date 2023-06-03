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

def process_sentences(sentences):
    for sent_no in range(len(sentences)):
        sentences[sent_no] = 'Research topic is'+'[SEP]'+sentences[sent_no]
    return sentences
    
def getEmbeddings(modelName, sentences):
    model = load_model(modelName)
    if modelName == "sentence-transformers/allenai-specter":
        sentences = process_sentences(sentences)
    deviceName = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {deviceName} to encode...")
    with torch.inference_mode():
        sentEmbedding = model.encode(sentences, device=deviceName)
        return sentEmbedding

def createEmbeddings(modelName, sentences, embeddingFileName):
    if os.path.exists(embeddingFileName):
        os.remove(embeddingFileName)
    h5Obj = h5py.File(embeddingFileName, 'w')
    sentEmbedding = getEmbeddings(modelName, sentences)
    embeddingDim = sentEmbedding.shape[1]
    h5Obj.create_dataset(f'researchIntEmbedding', data=sentEmbedding, chunks=True, maxshape=(None,embeddingDim))
    h5Obj.close()

#pre-trained model
#modelName = "sentence-transformers/all-mpnet-base-v2"
modelName = "sentence-transformers/allenai-specter"
#name of embedding file
researchIntEmbeddingFileName = "embeddings_"+"_".join(modelName.split('/'))+".h5"
#unique research interests
idToResearchIntDict = read_from_json("indices_to_researchInterest.json")
#sort items per indices in acsending order
idToResearchIntDictItems = sorted(list(idToResearchIntDict.items()), key=lambda x: int(x[0]))
researchIntList = [value for key, value in idToResearchIntDictItems]
#overwrite embedding file if it exists
execution_time = timeit.timeit(lambda: createEmbeddings(modelName, researchIntList, researchIntEmbeddingFileName), number=1)
print(f"Time taken to encode {len(researchIntList)} research interests is {execution_time} seconds i.e {execution_time/len(researchIntList)} seconds per item")
    
