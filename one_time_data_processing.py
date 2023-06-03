import json

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

def process_faculty_data(data):
    #process faculty data    
    uniqResearchInt = sorted(list(getUniqResearchInt(data)))
    #assigns an index to each research_interest availble in the data
    idToResearchIntDict = {idx: name for idx, name in enumerate(uniqResearchInt)}
    #save this data to do incremental maintenance
    write_to_json("indices_to_researchInterest.json", idToResearchIntDict)

    #assigns each faculty an index
    researchIntToIdDict = {name: idx for idx, name in enumerate(uniqResearchInt)}
    #save this data to do incremental maintenance
    write_to_json("researchInterest_to_indices.json", researchIntToIdDict)

file_name = "faculty_data.json"
faculty_data = read_from_json(file_name)
process_faculty_data(faculty_data)

