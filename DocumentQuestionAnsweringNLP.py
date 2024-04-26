# %%

# %%
#loading libraries
import pdfplumber
import requests
from transformers import pipeline, AutoTokenizer

#download file
def downloadFiles(url, name):
    response = requests.get(url)
    pdf_file = open(name + ".pdf", "wb")
    pdf_file.write(response.content)
    pdf_file.close()

# %%
#files with file url, name, page_number and device_name
files = []
files.append({"url": "http://www.ti.com/lit/ds/symlink/lm46000-q1.pdf", "name" : "LM46000QPWPTQ1", "page" : 46, "device_name"  : "LM46000-Q1"})
files.append({"url": "http://www.ti.com/lit/ds/symlink/lm46001-q1.pdf", "name" : "LM46001QPWPTQ1", "page" : 49, "device_name"  : "LM46001-Q1"})
files.append({"url": "http://www.ti.com/lit/ds/symlink/lm46002-q1.pdf", "name" : "LM46002QPWPTQ1", "page" : 48, "device_name"  : "LM46002-Q1"})
files.append({"url": "http://www.ti.com/lit/ds/symlink/lm5165.pdf", "name" : "LM5165XDRCT", "page" : 3, "device_name"  : "LM5165"})
files.append({"url": "http://www.ti.com/lit/ds/symlink/lm53600-q1.pdf", "name" : "LM53600AQDSXTQ1", "page" : 3, "device_name"  : "LM53600-Q1"})
files.append({"url": "http://www.ti.com/lit/ds/symlink/lm74610-q1.pdf", "name" : "LM74610QDGKTQ1", "page" : 25, "device_name"  : "LM74610-Q1"})

# %%
#querying model
def answerQuery(text, query, question_answerer):
    context = text
    result = question_answerer(question=query,context=context)
    return {"Answer" : result['answer'], "score": round(result['score'], 4), "start": result['start'], "end": result['end']}

# %%
#trimming result
def processResult(res, query):
    res.sort(key=lambda x: x['score'], reverse=True)
    result = {"query": query, "maxRes": max(res, key=lambda x: x['score']), "top10Res": res[:10]}
    return result

# %%
#process files
def processFile(pdf_file, num_pages, line_threshold, question_answerer, queries):
    text = ""
    results = []

    with pdfplumber.open(pdf_file) as pdf:
        pages = pdf.pages[0:num_pages]  #Only using first page in my instance
        for query in queries:
            result = []
            for page in pages:
                lines = page.extract_text().split("\n")
                text = ""
                line_thres = 1
                for line in lines:
                    text += line + "\n"
                    if line_thres == line_threshold:
                        line_thres = 1
                        result.append(answerQuery(text, query, question_answerer))
                        text = ""
                    else:
                        line_thres += 1
            
            results.append(processResult(result, query))
    return results

# %%
#load_model
question_answerer = pipeline("question-answering", model='google-bert/bert-large-uncased-whole-word-masking-finetuned-squad')


#params
line_threshold = 10     # this can be changed for fine tuning of results
num_pages = 6           # this can typically be set between 20 - 25 for more scanning

allResults = []

#path to .pdf file
for file in files:
    #queries
    queries = []
    queries.append("What is the Brand of " + file['device_name'] + "?")
    queries.append("What is the Regulator Function of " + file['device_name'] + "?")
    queries.append("What is the Input Voltage Range of " + file['device_name'] + "?")
    queries.append("What is the Output Voltage Range of " + file['device_name'] + "?")
    queries.append("What is the Maximum Output Current of " + file['device_name'] + "?")
    queries.append("What are the total Number of Outputs of " + file['device_name'] + "?")
    queries.append("What is the Mounting Type of " + file['device_name'] + "?")
    queries.append("What is the Package Type of " + file['device_name'] + "?")
    queries.append("How many pin device is " + file['device_name'] + "?")
    queries.append("What is the Output Type of " + file['device_name'] + "?")
    queries.append("What is the Maximum Switching Frequency of " + file['device_name'] + "?")
    #queries.append("What kind of Switching Regulator is in " + file['device_name'] + "?")
    queries.append("What is the Maximum Efficiency of " + file['device_name'] + "?")
    queries.append("What is the Operating Temperature Range of " + file['device_name'] + "?")
    queries.append("What are the Dimensions of " + file['device_name'] + "?")
    queries.append("What are the Package Dimensions of " + file['device_name'] + "?")
    queries.append("What is the Minimum Output Voltage of " + file['device_name'] + "?")
    queries.append("What is the Minimum Input Voltage of " + file['device_name'] + "?")
    queries.append("What is the Length of " + file['device_name'] + "?")
    queries.append("What is the Width of " + file['device_name'] + "?")
    queries.append("What is the Height of " + file['device_name'] + "?")
    queries.append("What is the Minimum Operating Temperature of " + file['device_name'] + "?")
    queries.append("What is the Maximum Operating Temperature of " + file['device_name'] + "?")
    queries.append("What is the Maximum Input Voltage of " + file['device_name'] + "?")
    queries.append("What is the Maximum Output Voltage of " + file['device_name'] + "?")

    downloadFiles(file['url'], file['name'])

    pdf_file = r"%s.pdf" % (file['name'])
    
    fileResult = processFile(pdf_file, num_pages, line_threshold, question_answerer, queries)
    allResults.append({"file" : file['name'], "result" : fileResult})


# %%
allResults

# %%
import json

# Convert the array to JSON format
json_array = json.dumps(allResults)

# Write the JSON to a file
with open('queries.json', 'w') as json_file:
    json_file.write(json_array)
    json_file.close()

# %%
import re

def getTempRange(item):
    temps = []
    top10res = item['top10Res']
    for res in top10res:
        # The regular expression pattern
        if "°C" in res['Answer']:
            if "to" in res['Answer']:
                temps.append({"Answer" : res['Answer'], "score" : res['score']})
    if len(temps) > 0:
        return max(temps, key=lambda x: x['score'])['Answer']
    else:
        return item['maxRes']['Answer']

def getVoltageRange(item):
    temps = []
    top10res = item['top10Res']
    for res in top10res:
        # The regular expression pattern
        if "V" in res['Answer']:
            if "to" in res['Answer']:
                temps.append({"Answer" : res['Answer'], "score" : res['score']})
    if len(temps) > 0:
        return max(temps, key=lambda x: x['score'])['Answer']
    else:
        return item['maxRes']['Answer']

def getVoltage(item):
    temps = []
    top10res = item['top10Res']
    for res in top10res:
        # The regular expression pattern
        if "V" in res['Answer']:
            temps.append({"Answer" : res['Answer'], "score" : res['score']})
    if len(temps) > 0:
        return max(temps, key=lambda x: x['score'])['Answer']
    else:
        return item['maxRes']['Answer']

def getMinTemperature(item):
    temps = []
    top10res = item['top10Res']
    for res in top10res:
        pattern = r'(?:.*?°C){1}'

        # Using re.search to find the pattern in the text
        match = re.match(pattern, res['Answer'])
        # The regular expression pattern
        if match:
            temps.append({"Answer" : res['Answer'], "score" : res['score']})
    if len(temps) > 0:
        ans = max(temps, key=lambda x: x['score'])['Answer']
        if "to" in ans:
            ans = ans.split('to')[0]
        return ans
    else:
        return item['maxRes']['Answer']

def getMaxTemperature(item):
    temps = []
    top10res = item['top10Res']
    for res in top10res:
        pattern = r'(?:.*?°C){1}'

        # Using re.search to find the pattern in the text
        match = re.match(pattern, res['Answer'])
        # The regular expression pattern
        if match:
            temps.append({"Answer" : res['Answer'], "score" : res['score']})
    if len(temps) > 0:
        ans = max(temps, key=lambda x: x['score'])['Answer']
        if "to" in ans:
            ans = ans.split('to')[1]
        return ans
    else:
        return item['maxRes']['Answer']
    
def getCurrent(item):
    temps = []
    top10res = item['top10Res']
    for res in top10res:
        # The regular expression pattern
        if "A" in res['Answer'] or "Ω" in res['Answer']:
            temps.append({"Answer" : res['Answer'], "score" : res['score']})
    if len(temps) > 0:
        return max(temps, key=lambda x: x['score'])['Answer']
    else:
        return item['maxRes']['Answer']

def getFrequency(item):
    temps = []
    top10res = item['top10Res']
    for res in top10res:
        # The regular expression pattern
        if "Hz" in res['Answer']:
            temps.append({"Answer" : res['Answer'], "score" : res['score']})
    if len(temps) > 0:
        ans = max(temps, key=lambda x: x['score'])['Answer']
        return re.match(r"(.*Hz)", ans).group()
    else:
        return item['maxRes']['Answer']

def getEfficiency(item):
    temps = []
    top10res = item['top10Res']
    for res in top10res:
        # The regular expression pattern
        if "%" in res['Answer']:
            temps.append({"Answer" : res['Answer'], "score" : res['score']})
    if len(temps) > 0:
        return max(temps, key=lambda x: x['score'])['Answer']
    else:
        return item['maxRes']['Answer']

def getDimensions(item):
    temps = []
    top10res = item['top10Res']
    for res in top10res:
        # The regular expression pattern
        pattern = r'(?:.*?mm){3}'

        # Using re.search to find the pattern in the text
        match = re.search(pattern, res['Answer'])
        
        if match:
            temps.append({"Answer" : res['Answer'], "score" : res['score']})
    if len(temps) > 0:
        return max(temps, key=lambda x: x['score'])['Answer']
    else:
        return item['maxRes']['Answer']

# %%
import json

manufacturer = []

attributes = []

values = []
with open('queries.json', 'r') as f:
    data = json.load(f)
    for item in data:
        attributes.append("Brand")
        attributes.append("Regulator Function")
        attributes.append("Input Voltage Range")
        attributes.append("Output Voltage Range")
        attributes.append("Maximum Output Current")
        attributes.append("Number of Outputs")
        attributes.append("Mounting Type")
        attributes.append("Package Type")
        attributes.append("Pin Count")
        attributes.append("Output Type")
        attributes.append("Maximum Switching Frequency")
        #attributes.append("Switching Regulator")
        attributes.append("Efficiency")
        attributes.append("Operating Temperature Range")
        attributes.append("Dimensions")
        attributes.append("Package Dimensions")
        attributes.append("Minimum Output Voltage")
        attributes.append("Minimum Input Voltage")
        attributes.append("Length")
        attributes.append("Width")
        attributes.append("Height")
        attributes.append("Minimum Operating Temperature")
        attributes.append("Maximum Operating Temperature")
        attributes.append("Maximum Input Voltage")
        attributes.append("Maximum Output Voltage")

        minMax = False
        for res in item['result']:
            manufacturer.append(item['file'])
            if "Operating Temperature Range" in res['query']:
                tempRange = getTempRange(res)
                if "to" in tempRange:
                    minMax = True
                    minTemp = tempRange.split("to")[0]
                    maxTemp = tempRange.split("to")[1]
                values.append(tempRange)
            elif "Voltage Range" in res['query']:
                values.append(getVoltageRange(res))
            elif "Output Current" in res['query']:
                values.append(getCurrent(res))
            elif "Maximum Switching Frequency" in res['query']:
                values.append(getFrequency(res))
            elif "Maximum Efficiency" in res['query']:
                values.append(getEfficiency(res))
            elif "Dimensions" in res['query']:
                values.append(getDimensions(res))
            elif "Output Voltage" in res['query']:
                values.append(getVoltage(res))
            elif "Input Voltage" in res['query']:
                values.append(getVoltage(res))
            elif "Minimum Operating Temperature" in res['query']:
                if minMax == True:
                    values.append(minTemp)
                else:
                    values.append(getMaxTemperature(res))
            elif "Maximum Operating Temperature" in res['query']:
                if minMax == True:
                    values.append(maxTemp)
                else:
                    values.append(getMaxTemperature(res))
            else:
                values.append(res['maxRes']['Answer'])


# %%
print(len(manufacturer))
print(len(attributes))
print(len(values))

# %%
import pandas as pd

df = pd.DataFrame(data={'Manufacturer Part Number' : manufacturer, 'Attribute' : attributes, 'Attribute Value' : values })

# %%
print(df.head(46))

# %%
df.to_excel("Assignment - AS-AIML-01 - Output File.xlsx", index=False)

# %%



