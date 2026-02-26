"""
methodes to compute the metrics that are used to evaluate the LLM

for the training we will mesure the following metrics:
 - related to CrossEntropy:
    - CrossEntropy (CE) [TRAIN]
        quantifies how uncertain a language model is when predicting the next token in a sequence. 
        (Sensitivity to outliers, depends on the tokenizer)
    - Perplexity (PPL) [TRAIN]
        like CE, quantifies uncertainty of the next token 
        -> interpreted as the effective number of choices the model has for the next token, averaged over the sequence.
        (Sensitivity to outliers, depends on the tokenizer)
    - Bit-Per-Character (BPC) [TRAIN]
        quantifies uncertainty of the next character (in text)
        (Sensitivity to outliers, INDEPENDENT of the tokenizer)
        formula = lossCE * NbTokens / (NbCharsInText * log(2))
    - Bit-Per-Token (BPT) [TRAIN]
        formula = lossCE/log(2) (so same pros/cons as CE)

 - language:
    - Bilingual Evaluation Understudy (BLEU) [NON UTILISE]
        Evaluates n-gram overlap (up to 4). Focuses on precision; penalizes brevity.
    - Recall-Oriented Understudy for Gisting Evaluation (ROUGE) [NON UTILISE]
        Evaluates the specified n-gram overlap. Focuses on recall.

 - others:
    - Learning Rate (LR) [TRAIN]
        it migth change during training with some scheduling
        
 - sur la validité du fichier:
    (pour le TRAIN, se base sur les SVG reconstitués)
    (pour le TEST, se baser sur le SVG final generé)
    - Well-formedness rate (XML validity) [TRAIN + TESTS]
        % de sorties qui sont du XML valide (parse sans erreur).
    - SVG validity rate [TRAIN + TESTS]
        % de sorties acceptées par un renderer SVG (ex : navigateur, librsvg).
    - Render success rate [TRAIN + TESTS]
        % de sorties qui produisent effectivement une image non vide.

 - analyses visuels:
    - SSIM (Structural Similarity Index) [TRAIN + EVAL(1?)]
        Sensible à la structure visuelle globale.
 
 - sur la structure du fichier.
    - Tag distribution similarity (TAGS) [TRAIN + EVAL(1?)]
        Compare les histogrammes (<path>, <rect>, <circle>…).

Ces métriques capturent si le modèle apprend la “grammaire SVG”.

NOTE: pour l'evaluation faire 2 methodes
 1) on commence avec un "startOfSequence" seulement
 2) on commence avec le debut d'un des svg du dataset
et pour les deux methodes on le laisse generer un document jusqu'a EndOfSequence ou la limite de tokens
(si on atteins la limite de tokens essayer de finir le SVG -> voir comment s'y prendre)

NOTE: pour les metrics de TRAIN sur les fichiers complets, 
    (/!\\ attention il vas falloir modifier le forward pour renvoyer la loss & les logits)
il faut reconstituer des fichiers a partir des fragments generés 
    (on pouras faire plusieurs samplings a partire des fichiers reconstitués)
 - methode 1) etant donné que les morceaux generés s'entrecroisent
    on vas prendre le premier block complet puis la seconde moitiée de tout les blocks suivants
    de cette façon chaque moitiée béneficie du contexte qui lui precede 
    |-----------|-----|-----|-----|-----|-----|-----|---  (la sequence a reconstituer)
    ############[[[[[[######[[[[[[######[[[[[[######      (les chuncks generés)
          [[[[[[######[[[[[[######[[[[[[######[[[[[[####  (les chuncks generés)
    legende: 
        - "[": premiere moitiée du chunk (non conservés)
        - "#": seconde moitiée du chunck (la partie conservée)
"""