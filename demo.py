from balinese_pos_tagger import PosTagger

tagger = PosTagger()
text = "Mangkin kantun masisa malih asiki sané ngranayang Baliné kasub, inggih punika adat utawi budayané."
result = tagger.tag(text)

print(result)