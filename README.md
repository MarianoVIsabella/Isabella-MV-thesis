# Isabella MV Thesis

This is the repo for the framework "EnglishToAce" that I've developed for my 3rd year thesis in Computer Science at Unical. 
Inside the repo there are several files: 
* This file, the README.md, which contains informations about the repo (in English); 
* The README.it.md file, which contains the same informations about the repo but in Italian;
* The file Citation.Cff which contains the informations useful to cite this repo (please follow it if you want to do so)
* Lastly, the file EnglishToAceV2.py, which is the code I've written.
## What does the framework do?

In the code inside the repo there are a couple of random sentences to test it, feel free to change them because they don't really matter in the workflow of the framework.<br>The goal of the project was to create a Python tool to convert standard English sentences in Attempto Controlled English(from now on, ACE) sentences. <br>
The framework implements a couple of ACE rules, here mentioned:
* Converts sentences in active sentences which respect the Subject Verb Object structure.
* Handles simple cases of saxon genitive.
* Converts passive sentences into active sentences.
* Splits the relative clauses in a sentence into separate ACE valid sentences.
* Processes object complements introduced by "that".
* By using WordNet, it's able to replace the word too complex with more general hypernyms.
* Converts the irregular plural.
* Normalizes time of the day according to ACE rules.
* Tracks the introduction of new objects, giving them the article "a" or "the" if the object was already introduced in the sentence.

## Which dependencies I need?
Other than the standard library "re", which doesn't need anything on its own, the main dependencies are [spaCy](https://spacy.io/) and WordNet, available in [NLTK](https://www.nltk.org/).
 
## What's the pipeline?

In the Python file there are some comments (in Italian) which describes more precisely what does the code do, but essentially the pipeline is as follows:
* _Parsing_ using spaCy in order to identify sentence structures, dependencies and named entities.
* _Cause splitting_ in order to divide a complex sentences into simpler clauses.
* _SVO Extraction_ in order to extract the subject, the verb and the object from each clause.
* _Simplification_ using WordNet in order to replace complex words with simpler synonyms or hypernyms.
* _Article Assignment_ in order to assign "a" or "the" to each noun phrase.
* _Sentence Generation_ in order to assemble the resulting ACE sentence(s).
