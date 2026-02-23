"""
methodes to compute the metrics that are used to evaluate the LLM

for the training we will mesure the following metrics:
 - related to CrossEntropy:
    - CrossEntropy (CE) 
        quantifies how uncertain a language model is when predicting the next token in a sequence. 
        (Sensitivity to outliers, depends on the tokenizer)
    - Perplexity (PPL)
        like CE, quantifies uncertainty of the next token 
        -> interpreted as the effective number of choices the model has for the next token, averaged over the sequence.
        (Sensitivity to outliers, depends on the tokenizer)
    - Bit-Per-Character (BPC)
        quantifies uncertainty of the next character (in text)
        (Sensitivity to outliers, INDEPENDENT of the tokenizer)
        formula = lossCE * NbTokens / (NbCharsInText * log(2))
    - Bit-Per-Token (BPT)
        formula = lossCE/log(2) (so same pros/cons as CE)
 - others:
    - Bilingual Evaluation Understudy (BLEU) 
        Evaluates n-gram overlap (up to 4). Focuses on precision; penalizes brevity.
    - Recall-Oriented Understudy for Gisting Evaluation (ROUGE)
        Evaluates the specified n-gram overlap. Focuses on recall.
    - Learning Rate (LR)
        it migth change during training with some scheduling
"""