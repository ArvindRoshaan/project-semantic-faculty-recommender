#imports for data processing/storing
import math
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

def display_faculty_info(col_obj, faculty_dict, research_list):
    col_obj.markdown(f"[![URL]({faculty_dict['Image URL']})]({faculty_dict['URL']})")

    name = faculty_dict['Name']
    name_string = f"[**{name}**]({faculty_dict['URL']})" if "URL" in faculty_dict else f"**{name}**"
    position_string = f"{faculty_dict['Position']}\n"

    has_degree = "Ph.D" in faculty_dict
    
    research_string = ""
    has_research_interest = "Research Interests" in faculty_dict
    num_research_int = len(faculty_dict['Research Interests']) if has_research_interest else 0
    other_research_interest = list(set(faculty_dict['Research Interests']).difference(set(research_list)))
    for i in range(min(3, num_research_int)):
        if i < len(research_list):
            research_string += f"- **{research_list[i]}**  \n"
        else:
            research_string += f"- {other_research_interest[i-len(research_list)]}  \n"
    

    col_obj.markdown(f"{name_string}")
    col_obj.markdown(f"{position_string}")
    if has_degree:
         col_obj.markdown(f"{faculty_dict['Ph.D']}")
    if has_research_interest:
         col_obj.markdown(f"{research_string}")

@st.cache_data
def load_css():
    with open('./files/style.css') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

def display_output(topIndices):
    facultyIdsList = []
    count_faculties = 0
    facultyIdsIdxInList = dict()
    uniqFacultyIds = set()
    researchIntIdsByFacultyOrder = []
    for i in range(len(topIndices)):
        researchIntId = topIndices[i]
        #similarityValue = topCosSim[i]
        facultyId = researchIntIdToFacultyIdDict[str(researchIntId)]
        for fId in facultyId:
            #print(fId, data[fId]['Name'], idToResearchIntDict[str(researchIntId)])
            if fId not in uniqFacultyIds:
                uniqFacultyIds.add(fId)
                facultyIdsList.append(fId)
                facultyIdsIdxInList[fId] = count_faculties
                researchIntIdsByFacultyOrder.append([idToResearchIntDict[str(researchIntId)]])
                count_faculties += 1
            else:
                researchIntIdsByFacultyOrder[facultyIdsIdxInList[fId]].append(idToResearchIntDict[str(researchIntId)])

    #for fId in facultyIdsList:
        #print(data[fId]['Name'], researchIntIdsByFacultyOrder[facultyIdsIdxInList[fId]])
    max_columns = 4
    num_rows = math.ceil(len(facultyIdsList)/4)
    col_objs = [st.columns(max_columns) for i in range(num_rows)]
    for i in range(len(facultyIdsList)):
        #print(math.floor(i/4), i%4)
        col_obj = col_objs[math.floor(i/4)][i%4]
        display_faculty_info(col_obj, data[facultyIdsList[i]], researchIntIdsByFacultyOrder[facultyIdsIdxInList[facultyIdsList[i]]])


#data file load
data = read_from_json("faculty_data.json")
#mapping file load
#index -> research interest mapping
idToResearchIntDict = read_from_json("indices_to_researchInterest.json")
#research interest index -> faculty  interest mapping
researchIntIdToFacultyIdDict = read_from_json("researchIntIndices_to_FacultyIndices.json")

#pre-trained model load
#modelName = "sentence-transformers/all-mpnet-base-v2"
modelName = "sentence-transformers/allenai-specter"
model = load_model(modelName)

#embeddings file load
embeddingFileName = "embeddings_"+"_".join(modelName.split('/'))+".h5"
embeddingsObj = load_embeddings(embeddingFileName)
embeddings = embeddingsObj['researchIntEmbedding']

st.title("Semantic Searchbar")
st.subheader("Search IIT Hyderabad faculties by their stated research interest")
query = st.text_input("Enter a research keyword below and get the IIT Hyderabad faculties with the most similar research interests")
   

if query != "":
    load_css()
    if modelName == "sentence-transformers/allenai-specter":
        queryList = process_sentences([query])
    print(f"The keyword is {query}")
    #top k research interests
    #topKResearchInterests = getTopK(queryList, np.array(embeddings), model, k=15)
    #topIndices = list(topKResearchInterests.indices.numpy()[0])
    #topCosSim = list(topKResearchInterests.values.numpy()[0])
    #research interests above a threshold
    topIndices, topCosSim = getMostSimilarSentEmbeddings(queryList, np.array(embeddings), model)
    if len(topIndices) == 0:
        st.info(f"None of the faculties at IIT Hyderabad have research interest closely matching **{query}**. Please try rephrasing the keyword...")
    else:
        display_output(topIndices)