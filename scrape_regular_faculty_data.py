import time
from tqdm import tqdm
import json

#imports to scrape data from a webpage
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def getAllRegularFacultyInfo(url):
    # make a request to the webpage
    response = requests.get(url)
    # use Beautiful Soup to parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    #returns the faculty info; we can access its element just like a list
    return soup.find_all(id='regular')[0].find_all(class_="facultycard")

#returns faculty designation
getPosition = lambda x: x.text.split(":")[1].strip()
#returns a list of departments that faculty is associated with
getDepartments = lambda y, idx: list(map(lambda x: x.text.strip(), y.find_all(style="margin:auto; padding:0 1vw;")[idx].find_all("li")))
#returns the highest degree obtatined (with college info)
getQualification = lambda x: x.text.strip()
#returns a list of research interests of the faculty
getResearchInt = lambda y, idx: list(map(lambda x: x.text.strip(), y.find_all(style="margin:auto; padding:0 1vw;")[idx].find_all("li")))
#return a formatted office address of the faculty
getOfficeAddress = lambda z: ", ".join(list(filter(lambda y: y!='', map(lambda x: x.strip(), z.find_all(style="margin:auto; padding:0 1vw; line-height: initial; font-size: 14px;")[0].find("p").text.split("\n"))))).replace(',,', ',')
#returns the college mail id of the faculty
getMail = lambda y: "".join(list(map(lambda x: chr(int(x[:2],16))+x[2:], y.find('a')['href'].split('%')[1:])))
#returns the office phone of the faculty
getOfficePhone = lambda y: y.text.split(':')[1].strip()
#returns the homepage of the faculty
getHomepage = lambda y: y.find('a')['href']

functions_map = {'Position' : getPosition,
                'Department(s)' : getDepartments,
                'Ph.D' : getQualification,
                'Research Interests' : getResearchInt,
                'Office Address' : getOfficeAddress,
                'E-mail' : getMail,
                'Office Phone' : getOfficePhone,
                'Homepage' : getHomepage}

def getFacultyInfo(info):
    faculty_dict = {}
    #get url of iith faculty webpage and url of faculty photo
    img_html = info.find('a')
    faculty_url = urljoin(home_url, img_html['href'])
    faculty_dict['URL'] = faculty_url
    faculty_dict['Image URL'] = urljoin(home_url, img_html.find('img')['src'])
    
    #get the contents from iith faculty url
    faculty_page = requests.get(faculty_url)
    faculty_page = BeautifulSoup(faculty_page.text,"html.parser")
    page_info = faculty_page.find(class_="col-sm-9 article-post")
    
    #get all available fields from the iith faculty url
    faculty_dict['Name'] = page_info.find("h3").text.strip()
    index_to_access_list_items = 0
    for h6Items in page_info.find_all("h6"):
        field = h6Items.text.split(":")[0].strip()         
        
        if field in {'Department(s)', 'Research Interests'}:
            faculty_dict[field] = functions_map[field](page_info, index_to_access_list_items)
            index_to_access_list_items += 1
        elif field in {'Office Address'}:
            faculty_dict[field] = functions_map[field](page_info)
        elif field in {'Position', 'Ph.D', 'E-mail', 'Office Phone', 'Homepage'}:
            faculty_dict[field] = functions_map[field](h6Items)
        else:
            pass
    #return all available attributes listed in the iith faculty url
    return faculty_dict

def writeToJSON(fileName, listOfDicts):
    print("Writting to file...")
    with open(fileName, 'w') as f:
        json.dump(listOfDicts, f)
        
def saveFacultyData(fileName, data):
    #accumulate info of all faculties
    faculty_data = []
    for idx in tqdm(range(len(data))):
        card = data[idx]
        faculty_dict = getFacultyInfo(card)
        faculty_data.append(faculty_dict)
        time.sleep(5)
        if idx%50 == 0:
            #write operation for every 50 faculty
            writeToJSON(fileName, faculty_data)

    #final write operation
    writeToJSON(fileName, faculty_data)

def assert_successful_scrapping(fileName, data):
    with open(outFileName, 'r') as f:
        listOfDicts = json.load(f)
    if len(listOfDicts) == len(data):
        print(f"[DONE] The faculty data is successfully scrapped ({len(listOfDicts)}/{len(data)})")
    else:
        print(f"[ERROR] The faculty data is NOT fully scrapped ({len(listOfDicts)}/{len(data)})")
    
home_url = 'https://iith.ac.in/people/faculty/'
all_regular_faculty_info = getAllRegularFacultyInfo(home_url)
outFileName = 'faculty_data.json'
saveFacultyData(outFileName, all_regular_faculty_info)
assert_successful_scrapping(outFileName, all_regular_faculty_info)
