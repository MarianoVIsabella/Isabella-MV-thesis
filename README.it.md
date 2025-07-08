# Isabella MV Thesis

Questa è la repository per il framework "EnglishToAce" che ho sviluppato per la mia tesi triennale in informatica presso l'Unical.
Nella repo ci sono diversi file:
* Il file README.md, che contiene queste stesse informazioni sulla repo ma in inglese;
* Questo file,README.it.md,che contiene informazioni sulla repo (in italiano);
* Il file Citation.Cff che contiene le informazioni utili per citare questa repo(seguite le indicazioni lì contenute se volete farlo);
* Il file EnglishToAceDataset.csv, che contiene un piccolo dataset costruito da me utilizzato per testare il framework;
* Infine, il file EnglishToAceV2.py, che contiene il codice che ho scritto.
  
## Cosa fa il framework?

Nel codice c'è anche un main di prova per testare il funzionamento del framework con le frasi del dataset. <br> L'obiettivo era di creare un tool Python per tradurre frasi dall'inglese all'Attempto Controlled English (d'ora in poi, ACE). <br>
Il framework implementa alcune regole di ACE, di seguito elencate:
* Converte le frasi in frasi attive che rispettano la struttura SVO (Soggetto Verbo Oggetto).
* Gestisce casi semplici di genitivo sassone.
* Converte frasi dalla forma passiva a quella attiva.
* Divide le proposizioni relative di una frase in più frasi ACE valide. 
* Converte complementi oggetto introdotti da "that".
* Grazie a WordNet, sostituisce le parole troppo specifiche con dei loro iperonimi più semplici.
* Converte i plurali irregolari.
* Normalizza gli orari secondo le regole ACE.
* Mappa l'introduzione di nuovi oggetti nella frase, dando loro l'articolo "a" o "the" se l'oggetto era già stato introdotto.

## Quali dipendenze ci sono?
Oltre alla libreria "re", che non introduce nuove dipendenze da sola, le dipendenze principali sono [spaCy](https://spacy.io/) e WordNet, contenuto all'interno di [NLTK](https://www.nltk.org/).
 
## Come funziona?

Nel file Python ci sono dei commenti (in italiano) che descrivono più dettagliatamente cosa fa il codice, ma sostanzialmente ogni frase viene tradotta passando per queste fasi:
* _Analisi_ usando spaCy per identificare le strutture interne della frase, le dipendenze e le entità nominate (nomi propri ecc...).
* _Divisione in proposizioni_ per dividere una frase complessa in più proposizioni semplici.
* _Semplificazione_ usando WordNet per sostituire parole complesse con dei loro sinonimi o iperonimi più semplici.
* _Estrazione SVO_ per estrarre soggetto, verbo e oggetto da ogni proposizione.
* _Generazione delle frasi_ per costruire la frase (o le frasi) ACE risultante/i.

