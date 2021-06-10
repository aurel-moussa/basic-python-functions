from ibm_watson import LanguageTranslatorV3

url_lt='' #add end point here
apikey_lt='' #add api key here
version_lt='2018-05-01' #add which version of the language translator you want here

#instantiate connection and object
authenticator = IAMAuthenticator(apikey_lt)
language_translator = LanguageTranslatorV3(version=version_lt,authenticator=authenticator)
language_translator.set_service_url(url_lt)

#check if it works
language_translator

#check which languages are available
from pandas import json_normalize
json_normalize(language_translator.list_identifiable_languages().get_result(), "languages")

#ask for translation
translation_response = language_translator.translate(\
    text=recognized_text, model_id='en-fr')

#chek if it works
translation_response

#get only the important results
translation=translation_response.get_result()

#check if it works
translation

#get just the text
french_translation =translation['translations'][0]['translation']
french_translation


