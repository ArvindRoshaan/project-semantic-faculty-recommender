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

data = read_from_json('faculty_data.json')
researchIntToIdDict = read_from_json("researchInterest_to_indices.json")
idToResearchIntDict = read_from_json("indices_to_researchInterest.json")

researchIntIdToFacultyId = dict()
for i in range(len(data)):
    if 'Research Interests' not in data[i]:
        continue
    researchInt = data[i]['Research Interests']
    researchIntId = list(map(lambda x: researchIntToIdDict[x], researchInt))
    for id in researchIntId:
        if id in researchIntIdToFacultyId:
            researchIntIdToFacultyId[id].append(i)
        else:
            researchIntIdToFacultyId[id] = [i]

write_to_json("researchIntIndices_to_FacultyIndices.json", researchIntIdToFacultyId)
