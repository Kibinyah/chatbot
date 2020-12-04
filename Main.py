import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import random
import json
from flask import Flask, render_template, request, url_for
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob

boolDecisionTree = False
boolReasoning = False
counter = 0
stemmer = LancasterStemmer()
symptomList = ["", "", ""]

words = []
labels = []
word_list = []
label_list = []

# Load the json file for the bot to read from
def openFile():
    with open("intents.json") as file:
        data = json.load(file)
    return data

#Converts all the patterns in the JSON file into 4 lists.
def storeInLists():
    global words, labels, word_list, label_list
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # tokenise each word in a sentence and returns a list
            tokenWords = nltk.word_tokenize(pattern)
            # adds each tokenised word to words.
            words.extend(tokenWords)
            # add all token words to x_list as a list.
            word_list.append(tokenWords)
            # add each tag name to the y_list
            label_list.append(intent["tag"])
            # Add unique tags to labels if they do not already exist in labels list.
            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    labels = sorted(labels)
    # Stem and remove duplicates in the words list
    words = [stemmer.stem(word.lower()) for word in words if word != "?"]
    words = sorted(list(set(words)))

#Creates training and testing sets for the neural network training.
def initialiseData():
    training = []
    testing = []
    testing_empty = [0 for _ in range(len(labels))]

    for x, list in enumerate(word_list):
        bag = []
        some_words = [stemmer.stem(w) for w in list]
        # Loops through all the words and appends 1 to the bag if word matches to the stemmed words
        # Appends 0 if the word does not match
        for w in words:
            if w in some_words:
                bag.append(1)
            else:
                bag.append(0)

        # Add the contents in bag to training
        training.append(bag)
        testing_row = testing_empty[:]
        #finds the index of the label in label_list to set the value of the same index of testing_row to 1.
        testing_row[labels.index(label_list[x])] = 1
        testing.append(testing_row)

    training = numpy.array(training)
    testing = numpy.array(testing)

    return [training, testing]

#Build neural network and trains it using the training and testing datasets
def trainingModel(training, testing):
    # Build neural network of 1 hidden layer
    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 10)
    net = tflearn.fully_connected(net, len(testing[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    #Trains the neural network with training and testing datasets
    model.fit(training, testing, n_epoch=500, batch_size=8, show_metric=True)
    return model

#Converts user's message to a list of 1s and 0s
def bagOfWords(input):
    # tokenize the user input and stem each word
    input_words = nltk.word_tokenize(input)
    input_words = [stemmer.stem(word.lower()) for word in input_words]

    bag = [0 for _ in range(len(words))]
    for s in input_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    return numpy.array(bag)

#Selects the chatbot response based on the user's message
def chat(input):
    global boolDecisionTree
    global boolReasoning
    #predicts the message through the model to get a list of probabilities
    results = model.predict([bagOfWords(input)])[0]
    #Chooses the highest probability from the list of probabilities.
    index = numpy.argmax(results)
    #Finds the associated tag in the list of labels with the index.
    tag = labels[index]

    #Runs decisionTree method based on tag or boolean value
    if tag == "start" or boolDecisionTree is True:
        boolDecisionTree = True
        #Expects specific tags to run decision Tree
        if tag == "no" or tag == "yes" or tag == "start":
            response = decisionTree(tag)
            return response
        else:
            boolDecisionTree = False
            return "I do not understand... Stopping Questions"

    #Runs reasoning method based on tag or boolean value
    elif tag == "cough" or tag == "fever" or tag == "fever+cough" or tag == "fever+severe" or tag == "cough+severe" or \
            tag == "cough+fever+severe" or boolReasoning is True:
        boolReasoning = True
        response = reasoning(tag)
        return response

    #Runs webSearch method based on tag
    elif tag == "search":
        response = webSearch(input)
        return str(response)
    else:
        #If the highest probability is higher than 0.7
        #returns a random response of the tag read from the intents file.
        if results[index] > 0.7:
            for t in data["intents"]:
                if t['tag'] == tag:
                    responses = t['responses']
            return random.choice(responses)
        else:
            return "I do not understand"

#Asks questions in a decision tree style.
def decisionTree(tag):
    global counter
    global boolDecisionTree
    feverQ = "Do you have a high fever?"
    coughQ = "Do you have consistent dry coughs?"
    severityQ = "Are your symptoms severe?"
    if tag == "start":
        return feverQ
    if counter == 0 and tag == "yes":
        counter += 5
        return coughQ
    elif counter == 0 and tag == "no":
        counter -= 5
        return coughQ
    elif counter == 5 and tag == "yes":
        counter += 2
        return severityQ
    elif counter == 5 and tag == "no":
        counter -= 2
        return severityQ
    elif counter == -5 and tag == "yes":
        counter += 2
        return severityQ
    elif counter == -5 and tag == "no":
        boolDecisionTree = False
        counter = 0
        return "You do not have coronavirus symptoms"
    elif counter == 3 and tag == "yes":
        boolDecisionTree = False
        counter = 0
        return "It may not be from coronavirus but call NHS landline 111 for more urgent help"
    elif counter == 3 and tag == "no":
        boolDecisionTree = False
        counter = 0
        return "Not likely to be coronavirus, stay in self isolation unless conditions worsen"
    elif counter == 7 and tag == "yes":
        boolDecisionTree = False
        counter = 0
        return "You have severe coronavirus symptoms, you must call the NHS landline 111 for further help"
    elif counter == 7 and tag == "no":
        boolDecisionTree = False
        counter = 0
        return "You have mild coronavirus symptoms, stay in self-isolation unless your conditions worsens and then call the NHS."
    elif counter == -3 and tag == "yes":
        boolDecisionTree = False
        counter = 0
        return "You may not have coronavirus but call NHS if conditions becomes unbearable"
    elif counter == -3 and tag == "no":
        boolDecisionTree = False
        counter = 0
        return "It is not likely you have coronavirus, stay in quarantine unless conditions worsen."
    else:
        boolDecisionTree = False
        counter = 0
        return "I do not understand... Stopping Questions"

#Asks questions based on what the user has said
def reasoning(tag):
    global boolReasoning
    global symptomList
    feverQ = "Do you have a high fever?"
    coughQ = "Do you have consistent dry coughs?"
    severityQ = "Are your symptoms severe?"
    answer = ""
    #Checks the tag and checks the boolean list
    #From the complete boolean list, outputs a unique response.
    if (tag == "yes" or tag == "severe") and (
            symptomList == [True, False, ""] or symptomList == [False, True, ""] or symptomList == [True, True, ""]):
        symptomList[2] = True
        if symptomList == [True, False, True]:
            answer = "Your condition may not be from coronavirus but call NHS landline 111 for more urgent help"
        elif symptomList == [False, True, True]:
            answer = "Your condition may not be from coronavirus but call NHS landline 111 for more urgent help"
        elif symptomList == [True, True, True]:
            answer = "You have severe coronavirus symptoms, you must call the NHS landline 111 for further help"
        symptomList = ["", "", ""]
        boolReasoning = False
        return answer
    elif (tag == "no" or tag == "notSevere") and (
            symptomList == [True, False, ""] or symptomList == [False, True, ""] or symptomList == [True, True, ""]):
        symptomList[2] = False
        if symptomList == [True, False, False]:
            answer = "It is not likely you have coronavirus, stay in quarantine unless conditions worsen."
        elif symptomList == [False, True, False]:
            answer = "It is not likely you have coronavirus, stay in quarantine unless conditions worsen."
        elif symptomList == [True, True, False]:
            answer = "You have mild coronavirus symptoms, stay in self-isolation unless your conditions worsens and then call the NHS."
        symptomList = ["", "", ""]
        boolReasoning = False
        return answer

    elif (tag == "yes" or tag == "fever") and symptomList == [True, "", True]:
        symptomList[1] = True
        if symptomList == [True, True, True]:
            answer = "You have severe coronavirus symptoms, you must call the NHS landline 111 for further help"
        symptomList = ["", "", ""]
        boolReasoning = False
        return answer

    elif (tag == "no" or tag == "noFever") and symptomList == [True, "", True]:
        symptomList[1] = False
        if symptomList == [True, False, True]:
            answer = "Your condition may not be from coronavirus but call NHS landline 111 for more urgent help"
        symptomList = ["", "", ""]
        boolReasoning = False
        return answer

    elif (tag == "yes" or tag == "cough") and symptomList == ["", True, True]:
        symptomList[0] = True
        if symptomList == [True, True, True]:
            answer = "You have severe coronavirus symptoms, you must call the NHS landline 111 for further help"
        symptomList = ["", "", ""]
        boolReasoning = False
        return answer

    elif (tag == "no" or tag == "noCough") and symptomList == ["", True, True]:
        symptomList[0] = False
        if symptomList == [False, True, True]:
            answer = "Your condition may not be from coronavirus but call NHS landline 111 if it becomes unbearable"
        symptomList = ["", "", ""]
        boolReasoning = False
        return answer

    #Checks the tag and lists and sets the associated index of boolean list to True or False
    elif (tag == "yes" or tag == "fever") and (symptomList == [False, "", ""] or symptomList == [True, "", ""]):
        symptomList[1] = True
        return severityQ
    elif (tag == "no" or tag == "noFever") and (symptomList == [False, "", ""] or symptomList == [True, "", ""]):
        symptomList[1] = False
        return severityQ
    elif (tag == "yes" or tag == "cough") and (symptomList == ["", False, ""] or symptomList == ["", True, ""]):
        symptomList[0] = True
        return severityQ
    elif (tag == "no" or tag == "noCough") and (symptomList == ["", False, ""] or symptomList == ["", True, ""]):
        symptomList[0] = False
        return severityQ
    elif tag == "cough":
        symptomList[0] = True
        return feverQ
    elif tag == "fever":
        symptomList[1] = True
        return coughQ
    elif tag == "fever+cough":
        symptomList[0] = True
        symptomList[1] = True
        return severityQ
    elif tag == "fever+severe":
        symptomList[1] = True
        symptomList[2] = True
        return coughQ
    elif tag == "cough+severe":
        symptomList[0] = True
        symptomList[2] = True
        return feverQ
    elif tag == "cough+fever+severe":
        boolReasoning = False
        return "You have severe coronavirus symptoms, you must call the NHS landline 111 for further help"
    else:
        return "Please answer the question correctly"

#Performs a search on the internet to find a description
def webSearch(userText):
    #splits the message to individual words
    words = userText.split()
    term = words[-1]
    #Appends the last term onto the end of the URL
    url = 'https://en.wikipedia.org/wiki/{0}'.format(term)
    #searches the URL and gets all data from the website.
    data = requests.get(url)
    soup = BeautifulSoup(data.text, 'html.parser')
    #Searches for all paragraphs in the website data.
    paragraphs = soup.find_all('p')
    #Selects only the first two paragraphs
    blob_1 = TextBlob(paragraphs[0].get_text())
    blob_2 = TextBlob(paragraphs[1].get_text())
    description = blob_1 + " " + blob_2
    return description


data = openFile()

storeInLists()

datasets = initialiseData()

model = trainingModel(datasets[0], datasets[1])

#Creates a flask instance
app = Flask(__name__)

#Returns the chatbot webpage.
@app.route("/")
def index():
    return render_template("website.html")

#Get method that reads the user's message and passes it into the chat function
@app.route("/get")
def getResponse():
    userText = request.args.get('msg')
    response = chat(userText)
    return str(response)

#Runs the Flask instance
if __name__ == "__main__":
    app.run()
