from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sacrebleu

# Strings
reference = ["the cat is on the mat"]
candidate = "the cat sat on the mat"

# NLTK BLEU
nltk_score = sentence_bleu([reference[0].split()], candidate.split(), 
                           smoothing_function=SmoothingFunction().method1)
print(f"NLTK BLEU: {nltk_score}")

# SacreBLEU
sacrebleu_score = sacrebleu.sentence_bleu(candidate, reference)
print(f"SacreBLEU: {sacrebleu_score.score}")




# def get_tree_sitter_language(lang):
#     if lang == "python":
#         from tree_sitter_languages import get_parser
#         return get_parser("python")
#     raise NotImplementedError(f"Tree-sitter parser for language {lang} is not implemented.")
# get_tree_sitter_language("python")


# import evaluate
# metric = evaluate.load("dvitel/codebleu")

# prediction = "def add ( a , b ) :\n return a + b"
# reference = "def sum ( first , second ) :\n return second + first"

from codebleu import calc_codebleu

prediction = "def add ( a , b ) :\n return a + b"
reference = "def sum ( first , second ) :\n return second + first"

result = calc_codebleu([reference], [prediction], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
print(result)
# {
#   'codebleu': 0.5537, 
#   'ngram_match_score': 0.1041, 
#   'weighted_ngram_match_score': 0.1109, 
#   'syntax_match_score': 1.0, 
#   'dataflow_match_score': 1.0
# }

# Manually set up the language library

# {
#   'codebleu': 0.5537, 
#   'ngram_match_score': 0.1041, 
#   'weighted_ngram_match_score': 0.1109, 
#   'syntax_match_score': 1.0, 
#   'dataflow_match_score': 1.0
# }
