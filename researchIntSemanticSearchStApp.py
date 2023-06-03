#imports for data processing/storing
import json
import h5py
import numpy as np

#imports for modelling
import torch
from sentence_transformers import SentenceTransformer, util

#imports for UI/UX
import streamlit as st

st.set_page_config(
    page_title="Semantic Search",
    page_icon="🔍",
    layout="wide")

@st.cache_data
def read_from_json(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    print(f"{len(data)} entries are read from {file_name}")
    return data

@st.cache_resource
def load_model(modelName):
    #device set as per availability
    deviceName = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(deviceName)
    with torch.no_grad():    
        #model = torch.load('model_cpu.pt')
        model = SentenceTransformer(modelName, device=deviceName)
        model.to(device)
        model.eval()
        return model

@st.cache_resource
def load_embeddings(embeddingFileName):
    #load embeddings
    embeddings = h5py.File(embeddingFileName, 'r')
    return embeddings

@st.cache_data
def process_sentences(sentences):
    for sent_no in range(len(sentences)):
        sentences[sent_no] = 'Research topic is'+'[SEP]'+sentences[sent_no]
    return sentences

@st.cache_resource
def getQueryEmbedding(query, _model):
    #device set as per availability
    deviceName = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(deviceName)
    with torch.inference_mode():
        queryEmbedding = model.encode(query, device=device)
        return queryEmbedding

@st.cache_resource
def getTopK(query, sentEmbeddings, _model, k=5):
    queryEmbedding = getQueryEmbedding(query, model)
    topK = torch.topk(util.cos_sim(queryEmbedding, sentEmbeddings), k)
    return topK

@st.cache_resource
def getMostSimilarSentEmbeddings(query, sentEmbeddings, _model, thresh=0.90):
    queryEmbedding = getQueryEmbedding(query, model)
    cosSim = util.cos_sim(queryEmbedding, sentEmbeddings)
    cosSim = cosSim.squeeze()
    cosSimObj = [(idx, x) for idx, x in enumerate(cosSim.numpy())]
    aboveThreshCosSim = list(filter(lambda x: x[1]>thresh, cosSimObj))
    aboveThreshCosSimSortedObj = sorted(aboveThreshCosSim, key=lambda x: -x[1])
    similarResearchIntIndices = [aboveThreshCosSimSortedObj[idx][0] for idx in range(len(aboveThreshCosSimSortedObj))]
    similarResearchIntCosSim = [aboveThreshCosSimSortedObj[idx][1] for idx in range(len(aboveThreshCosSimSortedObj))]
    return similarResearchIntIndices, similarResearchIntCosSim

#data file load
dataFileName = "faculty_data.json"
data = read_from_json(dataFileName)
#mapping file load
#index -> research interest mapping
idToResearchIntDict = read_from_json("indices_to_researchInterest.json")
#research interest -> index mapping
researchIntToIdDict = read_from_json("researchInterest_to_indices.json")

#pre-trained model load
#modelName = "sentence-transformers/all-mpnet-base-v2"
modelName = "sentence-transformers/allenai-specter"
model = load_model(modelName)

#embeddings file load
embeddingFileName = "embeddings_"+"_".join(modelName.split('/'))+".h5"
embeddingsObj = load_embeddings(embeddingFileName)
embeddings = embeddingsObj['researchIntEmbedding']

st.title("Semantic Searchbar")
st.subheader("Research interest of faculties in IIT Hyderabad")
query = st.text_input("Enter a research keyword below and get the most relevant research interests of IIT Hyderabad faculties")
if query != "":
    if modelName == "sentence-transformers/allenai-specter":
        queryList = process_sentences([query])
    print(f"The query is {query}")
    #top k research interests
    #topKResearchInterests = getTopK(queryList, np.array(embeddings), model, k=15)
    #topIndices = list(topKResearchInterests.indices.numpy()[0])
    #topCosSim = list(topKResearchInterests.values.numpy()[0])
    #research interests above a threshold
    topIndices, topCosSim = getMostSimilarSentEmbeddings(queryList, np.array(embeddings), model)
    for i in range(len(topIndices)):
        researchIntId = topIndices[i]
        similarityValue = topCosSim[i]
        st.write(f"The research interest with rank {i+1} is **{idToResearchIntDict[str(researchIntId)]}** and has a cosine-similarity score of {similarityValue} with {query}")
    if len(topIndices) == 0:
        st.info(f"None of the research interests in our database closely matches **{query}**. Please try rephrasing the query...")



