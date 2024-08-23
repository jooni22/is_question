from transformers import AutoTokenizer, AutoModelForTokenClassification
from spacy_alignments.tokenizations import get_alignments
import torch

tokenizer = AutoTokenizer.from_pretrained("hfunakura/en-bertsemtagger-gold")
model = AutoModelForTokenClassification.from_pretrained("hfunakura/en-bertsemtagger-gold")

# define the tagset
id2semtag = {"0": "@@UNK@@", "1": "PRO", "2": "CTC", "3": "INT", "4": "EMP", "5": "DEC", "6": "ITJ", "7": "GRE", "8": "NEC", "9": "PFT", "10": "IMP", "11": "HAP", "12": "ROL", "13": "MOY", "14": "PRG", "15": "HAS", "16": "CLO", "17": "MOR", "18": "DEF", "19": "BUT", "20": "YOC", "21": "PRI", "22": "EQU", "23": "SUB", "24": "APX", "25": "REL", "26": "XCL", "27": "CON", "28": "GPO", "29": "QUE", "30": "DIS", "31": "IST", "32": "COL", "33": "SCO", "34": "GRP", "35": "EXS", "36": "FUT", "37": "ENS", "38": "QUC", "39": "DOM", "40": "SST", "41": "NIL", "42": "COO", "43": "QUV", "44": "PST", "45": "UNK", "46": "EXT", "47": "NTH", "48": "LIT", "49": "ORG", "50": "EXG", "51": "REF", "52": "DOW", "53": "TOP", "54": "EPS", "55": "DXT", "56": "AND", "57": "UOM", "58": "ALT", "59": "POS", "60": "PRX", "61": "GEO", "62": "BOT", "63": "DEG", "64": "ART", "65": "PER", "66": "GPE", "67": "EFS", "68": "DST", "69": "LES", "70": "ORD", "71": "NOT", "72": "NOW", "-100": "@@PAD@@"}

class SemtaggerPipeline():
    def __init__(self, model, tokenizer, id2semtag):
        self.model = model
        self.tokenizer = tokenizer
        self.id2semtag = id2semtag
    def predict(self, text):
        # get alignments
        encoding_list = self.tokenizer(text, add_special_tokens=False)
        encoded_tokens = self.tokenizer.convert_ids_to_tokens(encoding_list["input_ids"])
        words = text.split(" ")
        alignments = get_alignments(encoded_tokens, words)[1]
        is_first_list = []
        for alignment in alignments:
            is_first_list += [1] + [0]*(len(alignment)-1)
        is_first = torch.tensor(is_first_list)
        # yield and extract predictions 
        encoding = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        logits = model(**encoding).logits
        preds = logits.argmax(-1)[0][is_first==1]
        pred_labels = [self.id2semtag[str(int(i))] for i in preds]
        result = [f"{word}/{label}" for word, label in zip(words,pred_labels)]
        return " ".join(result)

pipeline = SemtaggerPipeline(model, tokenizer, id2semtag)
pipeline.predict("Jim and Mary smiled and left .")