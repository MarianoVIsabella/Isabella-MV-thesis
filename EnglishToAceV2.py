import spacy
from nltk.corpus import wordnet as wn
import re
import csv
import requests

APE_URL = "http://attempto.ifi.uzh.ch/ws/ape/apews.perl"

def send_to_ape(ace_sentence):
    payload = {
        "text": ace_sentence
    }
    try:
        response = requests.post(APE_URL, data=payload, timeout=10)
        if response.status_code == 200:
            return response.text.strip()
        else:
            return f"[APE ERROR {response.status_code}]"
    except Exception as e:
        return f"[EXCEPTION] {e}"

#carico il modello: non c'è un effettivo miglioramento dei risultati usando modelli più grandi, quindi si preferisce il più piccolo per ottimizzare.
nlp = spacy.load("en_core_web_sm")

# Set che tiene traccia degli oggetti già introdotti nelle frasi (o insiemi di frasi) da convertire.
# Deve essere resettato ogni volta che si cambia contesto (diversi capitoli di un libro ecc...) per fare in modo di pulire gli oggetti riferiti.
introduced_entities_lemmas = set()
#dizionario necessario perché spaCy non riesce a riconoscere con costanza alcuni plurali irregolari, neanche l'utilizzo di modelli
#più grandi ha aiutato a risolvere il problema.
irregular_plural_to_singular = {
    "people": "person",
    "feet": "foot",
    "geese": "goose",
    "mice": "mouse",
    "teeth": "tooth",
    "women": "woman",
    "children": "child",
    # lista potenzialmente da espandere
}

#funzione che restituisce un iperonimo sufficientemente vicino alla parola di partenza usando wordnet
def get_hypernym(word):
    synsets = wn.synsets(word)
    best_hypernym = word
    best_similarity = 0
    for syn in synsets:
        for h in syn.hypernyms():
            sim = syn.path_similarity(h)
            #sotto i 0.5 troppo generico, ma sopra troppo specifico
            if sim and sim > best_similarity and sim > 0.45:
                best_hypernym = h.lemmas()[0].name().lower()
                best_similarity = sim
    return best_hypernym

#euristica che stima se una parola è o meno semplice
def is_simple_word(word):
    synsets = wn.synsets(word)
    if not synsets:
        return False
    depths = [s.min_depth() for s in synsets]
    avg_depth = sum(depths) / len(depths)
    min_depth_val = min(depths)
    #questi parametri sono modificabili per rendere più o meno grande l'insieme delle parole riconosciute come "semplici" dal framework
    return min_depth_val <= 8 and avg_depth <= 9

#funzione che riceve in input una parola e determina se questa parola è già idonea per ACE o meno
def simplify_word(word):
    synsets = wn.synsets(word)
    if not synsets or is_simple_word(word):
        return word
    if word.lower() in irregular_plural_to_singular:
        return irregular_plural_to_singular[word.lower()]
    return get_hypernym(word)

#funzione che converte una frase passiva in attiva con la struttura SVO (Soggetto, Verbo, Oggetto) restituendo i singoli token e l'esito della conversione
def convert_passive_to_active_svo(doc):
    for token in doc:
        if token.dep_ == "nsubjpass" and token.head.dep_ in {"auxpass", "ROOT"}:
            verb_passive = token.head.head if token.head.head != token.head else token.head
            
            # l'oggetto della frase attiva sarà il soggetto passivo
            obj_active_token = token
            
            # determino l'agente nella frase passiva
            subj_active_token = None
            for child in verb_passive.children:
                if child.dep_ == "agent":
                    for agent_pobj in child.children:
                        if agent_pobj.dep_ == "pobj":
                            subj_active_token = agent_pobj
                            break
                    if subj_active_token:
                        break
            
            # se non c'è un agente, il soggetto della frase attiva è "Someone"
            if not subj_active_token:
                return nlp("Someone")[0], verb_passive, extract_chunk_span(obj_active_token), True
            else:
                return subj_active_token, verb_passive, extract_chunk_span(obj_active_token), True
    return None, None, None, False

# funzione ricorsiva per aggiungere i figli ai token
def add_children_and_conj(t, chunk_tokens):
    for child in t.children:
        # include i modificatori standard e preposizioni (con sottotree)
        if child.dep_ in {"det", "amod", "nummod", "poss", "compound", "prep", "quantmod"}:
            chunk_tokens.add(child)
            if child.dep_ == "prep":
                for sub_t in child.subtree:
                    chunk_tokens.add(sub_t)
        # include le congiunzioni coordinanti (cc)
        elif child.dep_ == "cc":
            chunk_tokens.add(child)
            # trova il congiunto associato a questo CC
            for conj_token in t.children:
                if conj_token.dep_ == "conj" and conj_token.head == t and conj_token.i > child.i:
                    chunk_tokens.add(conj_token)
                    add_children_and_conj(conj_token, chunk_tokens) 
                    break 

#funzione per ricavare l'intero "chunk" relativo a un token (frasi nominali ecc)
def extract_chunk_span(token):
    if not token:
        return ""
    chunk_tokens= set()
    chunk_tokens.add(token)
    add_children_and_conj(token, chunk_tokens)
    # conversione set in lista e ordinamento per rispettare l'ordine originale delle parole
    sorted_chunk_tokens = sorted(list(chunk_tokens), key=lambda t: t.i)
    
    return " ".join([t.text for t in sorted_chunk_tokens]).strip()

#funzione per semplificare le parole di un chunk ed evitare i termini non validi all'interno di ACE
def simplify_chunk(text, is_subject_or_object=False):
    doc = nlp(text)
    simplified_parts = []
    
    nominal_heads = []

    # quantificatori indefiniti (che altrimenti non posso riconoscere) da convertire in a/the
    quantifiers_to_standardize = {"several", "many", "few", "some", "any", "most", "all", "each", "every"} 
    
    quantifier_or_article_added = False 

    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"}:
            if token.dep_ not in {"cc", "conj"} and not (token.head and token.head.dep_ == "prep"):
                nominal_heads.append(token)
            elif token.dep_ == "conj" and token.head and token.head.head and token.head.head.pos_ in {"NOUN", "PROPN"}:
                nominal_heads.append(token)
    
    if not is_subject_or_object or not nominal_heads:
        for t in doc:
            if t.pos_ in {"NOUN", "VERB", "ADJ", "PROPN"}:
                simplified_parts.append(simplify_word(t.lemma_.lower()))
            elif t.pos_ in {"DET", "NUM"}:
                if t.pos_ == "NUM" or t.dep_ == "nummod": 
                    simplified_parts.append(t.text)
                elif t.text.lower() not in {"a", "an", "the"} and t.text.lower() not in quantifiers_to_standardize:
                    simplified_parts.append(t.text)
                else: 
                    if t.text.lower() in quantifiers_to_standardize:
                        simplified_parts.append("a")
                    else:
                        simplified_parts.append(t.text.lower())
            else:
                simplified_parts.append(t.text)
        return " ".join(simplified_parts).strip()

    current_chunk_lemmas = {simplify_word(head.lemma_.lower()) for head in nominal_heads}
    is_new_introduction = not current_chunk_lemmas.issubset(introduced_entities_lemmas)
    
    for t in doc:
        current_word = t.text
        
        if (t.pos_ == "DET" or t.text.lower() in quantifiers_to_standardize) and not quantifier_or_article_added:
            if t.pos_ == "NUM" or t.dep_ == "nummod": 
                simplified_parts.append(current_word)
            elif t.text.lower() in quantifiers_to_standardize or t.text.lower() in {"a", "an", "the"}:
                if t.text.lower() == "the" and not is_new_introduction:
                    simplified_parts.append("the")
                elif is_new_introduction:
                    simplified_parts.append("a")
                else: 
                    simplified_parts.append("the")
            else: 
                simplified_parts.append(current_word) 
            quantifier_or_article_added = True
            continue 
        
        if t.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}:
            simplified_parts.append(simplify_word(t.lemma_.lower()))
        else:
            if not (t.pos_ == "DET" or t.text.lower() in quantifiers_to_standardize):
                simplified_parts.append(current_word)
    
    if is_new_introduction:
        introduced_entities_lemmas.update(current_chunk_lemmas)
                
    return " ".join(simplified_parts).strip()

#funzione che ricava la struttura Soggetto Verbo Oggetto di una qualsiasi frase 
def extract_svo(doc):
    subject, verb, obj = None, None, None
    be_verb, adj = None, None
    adverbial_modifiers = []
    negation_modifier = None
    aux_verb = None

    for token in doc:
        if token.dep_ == "nsubj" and subject is None:
            subject = token
        elif token.pos_ == "VERB" and verb is None:
            if token.dep_ not in {"xcomp", "ccomp", "advcl", "conj", "acl"}:
                verb = token
        elif token.dep_ in {"dobj", "attr", "pobj"} and obj is None and token.pos_ in {"NOUN", "PROPN", "PRON", "ADJ"}:
            obj = token
        elif token.pos_ == "AUX" and token.lemma_ == "be" and be_verb is None:
            be_verb = token
        elif token.pos_ == "AUX" and token.lemma_ in {"do", "did", "does"} and aux_verb is None:
            aux_verb = token
        elif token.dep_ in {"acomp", "attr"} and adj is None:
            adj = token
        elif token.dep_ in {"advmod", "npadvmod", "prt"}:
            adverbial_modifiers.append(token)
        elif token.dep_ == "neg" and negation_modifier is None:
            negation_modifier = token
    #controlli fatti alla fine del ciclo per gestire casi particolari (frasi con il verbo essere in cui il verbo è appunto "to be" e l'oggetto è in realtà un aggettivo) 		
    if verb is None and be_verb:
        verb = be_verb
    if obj is None and adj:
        obj = adj

    return subject, verb, obj, adverbial_modifiers, negation_modifier, aux_verb

#funzione che si assicura i verbi abbiano sempre la coniugazione corretta
def conjugate_verb(verb_lemma, subject_token):
    
    is_subject_plural_for_conjugation = False
    #l'idea è verificare se il sogetto nella frase finale è singolare o plurale, e in base al soggetto coniugare.
    
    if subject_token.lemma_.lower() in ["we", "you", "they"]:
        is_subject_plural_for_conjugation = True
    
    elif subject_token.lemma_.lower() == "i":
        is_subject_plural_for_conjugation = True
    
    subj_chunk_doc = nlp(extract_chunk_span(subject_token))
    for t in subj_chunk_doc:
        if t.dep_ == "nummod" and t.pos_ == "NUM" and t.lemma_ not in ["one", "a", "an"]:
            is_subject_plural_for_conjugation = True
            break
    
    #gestico il verbo be
    if verb_lemma.lower() == "be":
        if is_subject_plural_for_conjugation:
            return "are"
        elif subject_token.lemma_.lower() == "i": 
            return "am"
        else:
            return "is"
    #gestisco il verbo have
    elif verb_lemma.lower() == "have":
        if is_subject_plural_for_conjugation:
            return "have"
        else:
            return "has"
    
    #se singolare, applico la coniugazione
    if is_subject_plural_for_conjugation:
        return verb_lemma 
    else:
        if verb_lemma.endswith(('s', 'sh', 'ch', 'x', 'z', 'o')):
            return verb_lemma + 'es'
        elif verb_lemma.endswith('y') and len(verb_lemma) > 1 and verb_lemma[-2] not in "aeiou":
            return verb_lemma[:-1] + 'ies'
        else:
            return verb_lemma + 's'

#funzione che semplifica le frasi singole in input
def simplify_sentence(text): 

    original_doc = nlp(text) 
    for token in original_doc:
        if token.dep_ == "poss" and token.head and token.head.pos_ in {"NOUN", "PROPN"}:
            possessor_token = token
            possessed_token = token.head
            
            possessor_text = extract_chunk_span(possessor_token)
            
            # ottengo il chunk del posseduto senza il possessore
            possessed_tokens_set = set()
            possessed_tokens_set.add(possessed_token)
            add_children_and_conj(possessed_token, possessed_tokens_set)
            possessed_tokens_set.discard(possessor_token)
            sorted_possessed_tokens = sorted(list(possessed_tokens_set), key=lambda t: t.i)
            possessed_text = " ".join([t.text for t in sorted_possessed_tokens]).strip()
            
            simplified_possessor = simplify_chunk(possessor_text, is_subject_or_object=True).capitalize()
            simplified_possessed = simplify_chunk(possessed_text, is_subject_or_object=True)
            
            # se non inizia già con un determinante, aggiungo "a" (ACE richiede un articolo)
            possessed_doc = nlp(simplified_possessed)
            first_token = possessed_doc[0] if possessed_doc else None
            if first_token and first_token.pos_ != "DET":
                simplified_possessed = f"a {simplified_possessed}"
            
            sentence = f"{simplified_possessor} has {simplified_possessed}."
            return sentence

    simplified_sentences=[]
    clauses = []
    
    #per prima cosa identifico se la frase può essere "scomposta" in più sotto-frasi
    root_token = None
    for token in original_doc:
        if token.dep_ == "ROOT":
            root_token = token
            break
            
    if root_token is None:
        clauses.append(list(original_doc))
    else:
        # trovo tutti i conjuncts coordinati per quella sotto-frase
        conjunct_roots = [root_token]
        
        for child in root_token.children:
            if child.dep_ == "conj" and child.pos_ == "VERB":
                conjunct_roots.append(child)
        
        # ristabilisco l'ordine corretto
        conjunct_roots.sort(key=lambda t: t.i)
        
        # calcolo gli indici a cui dividere le frasi
        clause_starts = [root.i for root in conjunct_roots]

        start_index = 0
        for root_i in clause_starts:
            end_index = original_doc[-1].i + 1
            if root_i + 1 < len(original_doc):
                for token in original_doc[root_i + 1:]:
                    if token.dep_ == "cc" and token.head == root_token:
                        end_index = token.i
                        break
                    
            clause_span = original_doc[start_index:end_index]
            clauses.append(list(clause_span))
            # la sotto-frase successiva inizierà all'indice successivo
            start_index = end_index + 1
        
        # aggiungo la parte conclusiva della frase originale eventualmente rimasta fuori all'ultima sotto-frase
        if start_index < len(original_doc):
            clauses.append(list(original_doc[start_index:]))
    # evito sotto-frasi vuote
    clauses = [c for c in clauses if any(t.text.strip() for t in c)]
    for clause_tokens in clauses:
        clause_text = " ".join([t.text for t in clause_tokens]).strip()
        if not clause_text:
            continue
            
        clause_doc = nlp(clause_text)

        active_subj_token, active_verb_token, active_obj_str, is_passive_converted = convert_passive_to_active_svo(clause_doc)
    
        if is_passive_converted:
            simplified_subj = simplify_chunk(extract_chunk_span(active_subj_token), is_subject_or_object=True).capitalize()
            conjugated_verb = conjugate_verb(active_verb_token.lemma_, active_subj_token)
            simplified_sentences.append(f"{simplified_subj} {conjugated_verb} {simplify_chunk(active_obj_str, is_subject_or_object=True)}.")
            continue

        relcl_info = []
        tokens_to_exclude_indices = set()

        for token in clause_doc:
            if token.dep_ == "relcl":
                rel_verb_token = token
                rel_subject_token = rel_verb_token.head
                relcl_info.append((rel_subject_token, rel_verb_token))
                
                for t in rel_verb_token.subtree:
                    tokens_to_exclude_indices.add(t.i)
                if rel_subject_token.dep_ == "relcl":
                    tokens_to_exclude_indices.add(rel_subject_token.i)

        for rel_subject_token, rel_verb_token in relcl_info:
            antecedent_text = extract_chunk_span(rel_subject_token)
            simplified_antecedent = simplify_chunk(antecedent_text, is_subject_or_object=True)
            
            rel_clause_span = list(rel_verb_token.subtree)
            rel_clause_doc = nlp(" ".join([t.text for t in rel_clause_span]))
            active_subj, active_verb, active_obj_str, is_passive_conv = convert_passive_to_active_svo(rel_clause_doc)
            
            if is_passive_conv:
                # forzo Someone come soggetto se active_subj è None o "who"
                if not active_subj or active_subj.lemma_.lower() in {"who", "which", "that"}:
                    active_subj_text = "Someone"
                else:
                    active_subj_text = simplify_chunk(extract_chunk_span(active_subj), is_subject_or_object=True).capitalize()
                
                # includo modificatori preposizionali collegati al verbo
                prepositional_phrases = []
                for child in rel_verb_token.children:
                    if child.dep_ in {"prep", "advmod", "npadvmod"}:
                        prep_tokens = set()
                        for t in child.subtree:
                            prep_tokens.add(t)
                        sorted_prep_tokens = sorted(list(prep_tokens), key=lambda t: t.i)
                        prep_phrase = " ".join([t.text for t in sorted_prep_tokens]).strip()
                        prepositional_phrases.append(simplify_chunk(prep_phrase))

                
                simplified_obj = simplified_antecedent
                conjugated_verb = conjugate_verb(active_verb.lemma_, nlp(active_subj_text)[0])
                
                sentence_parts = [active_subj_text, conjugated_verb, simplified_obj]
                if prepositional_phrases:
                    sentence_parts.append(" ".join(prepositional_phrases))
                
                simplified_sentence = " ".join(sentence_parts).strip() + "."
            else:
                rel_object_token = None
                for child in rel_verb_token.children:
                    if child.dep_ in {"dobj", "attr", "pobj"} and child.pos_ in {"NOUN", "PROPN", "PRON", "ADJ"}:
                        rel_object_token = child
                        break
                rel_object_text = extract_chunk_span(rel_object_token) if rel_object_token else ""
                simplified_rel_object = simplify_chunk(rel_object_text, is_subject_or_object=True)
                
                prepositional_phrases = []
                for child in rel_verb_token.children:
                    if child.dep_ in {"prep", "advmod", "npadvmod"}:
                        prep_tokens = set()
                        for t in child.subtree:
                            prep_tokens.add(t)
                        sorted_prep_tokens = sorted(list(prep_tokens), key=lambda t: t.i)
                        prep_phrase = " ".join([t.text for t in sorted_prep_tokens]).strip()
                        prepositional_phrases.append(simplify_chunk(prep_phrase))

                
                conjugated_rel_verb = conjugate_verb(rel_verb_token.lemma_, rel_subject_token)
                sentence_parts = [simplified_antecedent.capitalize(), conjugated_rel_verb, simplified_rel_object]
                if prepositional_phrases:
                    sentence_parts.append(" ".join(prepositional_phrases))
                
                simplified_sentence = " ".join(sentence_parts).strip() + "."
            
            simplified_sentences.append(simplified_sentence)
        # gestione delle proposizioni oggettive (ccomp, xcomp) collegate al verbo principale
        for token in clause_doc:
            if token.dep_ in {"ccomp", "xcomp"} and token.head and token.head.pos_ == "VERB":
                main_verb = token.head
                main_subject = None
                for child in main_verb.children:
                    if child.dep_ == "nsubj":
                        main_subject = child
                        break
                if not main_subject:
                    main_subject = nlp("Someone")[0]
                
                simplified_main_subject = simplify_chunk(extract_chunk_span(main_subject), is_subject_or_object=True).capitalize()
                conjugated_main_verb = conjugate_verb(main_verb.lemma_, main_subject)
                
                # processa la subordinata come frase secondaria
                comp_subtree = list(token.subtree)
                comp_text = " ".join([t.text for t in comp_subtree]).strip()
                comp_doc = nlp(comp_text)
                comp_subject, comp_verb, comp_obj, comp_adverbs, comp_neg, _ = extract_svo(comp_doc)
                
                if comp_subject and comp_verb:
                    simplified_comp_subj = simplify_chunk(extract_chunk_span(comp_subject), is_subject_or_object=True)
                    conjugated_comp_verb = conjugate_verb(comp_verb.lemma_, comp_subject)
                    simplified_comp_obj = ""
                    if comp_obj:
                        simplified_comp_obj = simplify_chunk(extract_chunk_span(comp_obj), is_subject_or_object=True)
                    
                    adverbs_str = " ".join([t.text for t in comp_adverbs])
                    if adverbs_str:
                        conjugated_comp_verb = f"{conjugated_comp_verb} {adverbs_str}"
                    
                    prepositional_phrases = []
                    for child in comp_verb.children:
                        if child.dep_ in {"prep", "advmod", "npadvmod"}:
                            prep_tokens = set(child.subtree)
                            sorted_prep_tokens = sorted(list(prep_tokens), key=lambda t: t.i)
                            prep_phrase = " ".join([t.text for t in sorted_prep_tokens]).strip()
                             #implemento con regex perché spacy da solo non mi da abbastanza informazioni per convertire l'orario 
                             #in formato valido per ACE
                            time_match = re.search(r'\b(\d{1,2})(?::(\d{2}))?\s*(AM|PM|am|pm)\b', prep_phrase)
                            if time_match:
                                hour = int(time_match.group(1))
                                minute = int(time_match.group(2)) if time_match.group(2) else 0
                                ampm = time_match.group(3).lower()
            
                                if ampm == "pm":
                                    hour = (hour+ 12) % 24
            
                                ace_time = f"{hour}:{minute:02d}"
                                prep_phrase = f"at {ace_time}"
                            prepositional_phrases.append(simplify_chunk(prep_phrase))
                    
                    comp_parts = [simplified_comp_subj, conjugated_comp_verb, simplified_comp_obj]
                    if prepositional_phrases:
                        comp_parts.append(" ".join(prepositional_phrases))
                    
                    simplified_comp_clause = " ".join([p for p in comp_parts if p]).strip()
                else:
                    simplified_comp_clause = "something occurs"
                
                simplified_sentence = f"{simplified_main_subject} {conjugated_main_verb} that {simplified_comp_clause}."
                simplified_sentences.append(simplified_sentence)



        main_clause_tokens = []
        for token in clause_doc:
            if token.i not in tokens_to_exclude_indices:
                main_clause_tokens.append(token.text)
        
        reconstructed_main_text = " ".join(main_clause_tokens).strip()
        
        main_doc_for_svo = nlp(reconstructed_main_text)
        
        subject, verb, obj, adverbial_modifiers, negation_modifier, aux_verb = extract_svo(main_doc_for_svo)
        
        effective_subjects_for_sentence_generation = []
        if subject:
            effective_subjects_for_sentence_generation.append(subject)
        
        effective_objects_for_sentence_generation = []
        if obj:
            effective_objects_for_sentence_generation.append(obj)

        if not effective_subjects_for_sentence_generation and verb:
            effective_subjects_for_sentence_generation = [nlp("Someone")[0]]

        if effective_subjects_for_sentence_generation and (effective_objects_for_sentence_generation or (verb and not effective_objects_for_sentence_generation)) and verb:
            s_token = effective_subjects_for_sentence_generation[0]
            subj_text = extract_chunk_span(s_token)
            simplified_s = simplify_chunk(subj_text, is_subject_or_object=True)
            conjugated_v = conjugate_verb(verb.lemma_, s_token)
            simplified_o = ""
            if effective_objects_for_sentence_generation:
                o_token = effective_objects_for_sentence_generation[0]
                obj_text = extract_chunk_span(o_token)
                simplified_o = simplify_chunk(obj_text, is_subject_or_object=True)

            adverbs_str = " ".join([t.text for t in adverbial_modifiers])
            if adverbs_str:
                conjugated_v = f"{conjugated_v} {adverbs_str}"

            if negation_modifier:
                affirmative_s = simplified_s.capitalize()
                affirmative_v = conjugated_v
                if not simplified_o and verb and verb.pos_ != "AUX" and verb.dep_ != "ROOT":
                    affirmative_o = "something"
                else:
                    affirmative_o = simplified_o
                
                affirmative_sentence_content = f"{affirmative_s} {affirmative_v} {affirmative_o}".strip()
                if affirmative_sentence_content.endswith("."):
                    affirmative_sentence_content = affirmative_sentence_content[:-1]
                    
                sentence = f"It is false that {affirmative_sentence_content.lower()}." 
            else:
                sentence = f"{simplified_s.capitalize()} {conjugated_v} {simplified_o}".strip()
                if not sentence.endswith("."):
                    sentence += "."

            if sentence.lower() not in [s.lower() for s in simplified_sentences]:
                simplified_sentences.append(sentence)
        elif effective_subjects_for_sentence_generation and verb:
                s_token = effective_subjects_for_sentence_generation[0]
                subj_text = extract_chunk_span(s_token)
                simplified_s = simplify_chunk(subj_text, is_subject_or_object=True)
                conjugated_v = conjugate_verb(verb.lemma_, s_token)
                simplified_o = ""

                adverbs_str = " ".join([t.text for t in adverbial_modifiers])
                if adverbs_str:
                    conjugated_v = f"{conjugated_v} {adverbs_str}"
                    simplified_o = "" 
                
                if negation_modifier:
                    affirmative_s = simplified_s.capitalize()
                    affirmative_v = conjugated_v
                    
                    affirmative_sentence_content = f"{affirmative_s} {affirmative_v}".strip()
                    if affirmative_sentence_content.endswith("."):
                        affirmative_sentence_content = affirmative_sentence_content[:-1]
                    sentence = f"It is false that {affirmative_sentence_content.lower()}."
                else:
                    sentence = f"{simplified_s.capitalize()} {conjugated_v} {simplified_o}".strip()
                    if not sentence.endswith("."):
                        sentence += "."

                if sentence.lower() not in [s.lower() for s in simplified_sentences]:
                    simplified_sentences.append(sentence)
        elif not simplified_sentences:
            
            if negation_modifier:
                simplified_sentences.append("It is false that something occurs.")
            else:
                simplified_sentences.append("Someone does something.")


    final_sentences = []
    seen_sentences_lower = set()
    for s in simplified_sentences:
        stripped_s = s.strip()
        stripped_s_clean = stripped_s
        if stripped_s_clean.endswith("."):
            stripped_s_clean = stripped_s_clean[:-1]
        if stripped_s_clean.lower() not in seen_sentences_lower and stripped_s_clean != "":
            if not stripped_s.endswith("."):
                stripped_s += "."
            final_sentences.append(stripped_s)
            seen_sentences_lower.add(stripped_s_clean.lower())

    return " ".join(final_sentences)

#processo un insieme di frasi con lo stesso contesto
def process_document_for_ace(sentences_list):
    final_document_sentences=[]
    for s in sentences_list:
        final_document_sentences.append(simplify_sentence(s))
    return " ".join(final_document_sentences)

if __name__ == "__main__":
    ace_valid_sentences=0
    ace_wrong_sentences=0
    with open("EnglishToACeDataset.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            introduced_entities_lemmas = set() 
            ace_sentence = simplify_sentence(row[0])
            if row[3] == ace_sentence:
                ace_valid_sentences+=1
                print(f"Original: {row[0]}\nACE: {ace_sentence}\n")
            else:
                ape_response = send_to_ape(ace_sentence)
                if "error" not in ape_response:
                    ace_valid_sentences+=1
                    print(f"Original: {row[0]}\nACE: {ace_sentence}\n")
                else:
                    ace_wrong_sentences+=1
            
    print (f"Le frasi valide per la logica ACE sono: {ace_valid_sentences}\n")
    print (f"Le frasi errate per ACE sono: {ace_wrong_sentences}")