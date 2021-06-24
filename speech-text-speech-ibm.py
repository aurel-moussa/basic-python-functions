"""
Simple functions to convert an audio request via speech-to-text, find response for this request via Watson Assistant and output response via text-to-speech
"""

#Install required packages ibm_watson and beautiful soup
!pip install PyJWT==1.7.1
!pip install ibm_watson bs4

#Load required packages
import os
from glob import glob
from bs4 import BeautifulSoup
import IPython
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1
from ibm_watson import AssistantV2
from ibm_watson import TextToSpeechV1

#os - to run commands in the environment via "os.popen".
#glob.glob - to find audio files.
#bs4 - to extract text from HTML.
#IPython - to play audio from Watson Text to Speech from within the notebook.
#ibm_cloud_sdk_core.authenticators.IAMAuthenticator - to help with API Key-based authentication.
#ibm_watson:
#SpeechToTextV1 - the Speech to Text service wrer.
#AssistantV2 - The Assistant service wrer.
#TextToSpeechV1 - the Text to Speech service wrer.

#SPEECH-TO-TEXT

recognition_service = SpeechToTextV1(IAMAuthenticator('{YOUR_APIKEY}')) #Instantiate a SpeechtoText; Watson API Key for Speech-to-Text goes here
recognition_service.set_service_url('{YOUR_ENDPOINT}') #Set service url; endpoint for Speech-to-Text goes here
SPEECH_EXTENSION = "*.webm" #the extension of the audio files that Speech to Text will need to analyze
SPEECH_AUDIOTYPE = "audio/webm" #the type of audio that Speech to Text will analyze. Watson supports webm, mp3, wav, ogg, and others

def recognize_audio():
    while len(glob(SPEECH_EXTENSION)) == 0: #whilere there is no audio file
        pass #null
    filename = glob(SPEECH_EXTENSION)[0] #filename of audio file
    audio_file = open(filename, "rb") #open file
    os.popen("rm " + filename)
    result = recognition_service.recognize(audio=audio_file, content_type=SPEECH_AUDIOTYPE).get_result() #sent file to Watson and request speech-to-text
    return result["results"][0]["alternatives"][0]["transcript"] #return results
    #"["results"][0]" - this will get the first set of results from Watson's response.
    #"["alternatives"][0]" - of all the alternative transcriptions, it'll get the first (most likely) one.
    #"["transcript"]" - of all the data Watson returns, only take the transcript string ("str" type in Python).
    
#TEXT-TO-ASSISTANT    

assistant = AssistantV2(version='2019-02-28', authenticator=IAMAuthenticator('{YOUR_APIKEY}')) #instantiate Watson Assistant, add API key
assistant.set_service_url('{YOUR_ENDPOINT}') #set service url, endpoint for Watson Assistant goes here
ASSISTANT_ID = "{YOUR_ASSISTANT_ID}" #add specific assistant that should be called
session_id = assistant.create_session(assistant_id =ASSISTANT_ID).get_result()["session_id"] #create new to keep track of context of convers

def message_assistant(text):
    response = assistant.message(assistant_id=ASSISTANT_ID,
                                 session_id=session_id,
                                 input={'message_type': 'text', 'text': text}).get_result() #give Assistant specific message and ask for response
    return BeautifulSoup(response["output"]["generic"][0]["text"]).get_text() #return response
  
#TEXT-TO-SPEECH
synthesis_service = TextToSpeechV1(IAMAuthenticator('{YOUR_APIKEY}')) #instatiate Text-to-Speech, add API key
synthesis_service.set_service_url('{YOUR_ENDPOINT}') #set service url, endpoint for T2S goes here

def speak_text(text):
    with open('temp_reponse.wav', 'wb') as audio_file: #create temporary audio file, "with" open in order to auto-close afterwards
        response = synthesis_service.synthesize(text, accept='audio/wav', voice="en-GB_JamesV3Voice").get_result() #ask for synthesis
        audio_file.write(response.content) #write Watson response into audio file
    return IPython.display.Audio("temp_response.wav", autoplay=True) #play audio file

#try it out
speak_text(message_assistant(recognize_audio()))
