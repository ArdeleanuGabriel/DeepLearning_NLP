import nltk
from nltk import CFG
from nltk.parse import RecursiveDescentParser

# Define the grammar
grammar = CFG.fromstring("""
    S -> NP VP | VP VP
    NP -> Adj Noun | unit unit unit | unit Verb NP | unit Verb | unit unit 
    VP -> Ceva Adj | Verb Noun | Verb Adv | Adv Verb | Comp Comp unit | NP Comp Comp unit | unit Verb Adv | unit Adv Verb
    Ceva -> Adv Verb
    unit -> temp Con Noun | Con Noun
    Con -> 'the' 
    Comp -> 'more' | 'than'
    temp -> 'of' | 'and'
    Noun -> 'planes' | 'parents' | 'bride' | 'groom'
    Verb -> 'be' | 'flying' | 'were' | 'loves'
    Adj -> 'dangerous' | 'flying' 
    Adv -> 'can' | 'were' | 'flying'
""")

parser = RecursiveDescentParser(grammar)

sentence1 = "flying planes can be dangerous".split()
sentence2 = "the parents of the bride and the groom were flying".split()
sentence3 = "the groom loves dangerous planes more than the bride".split()

for tree in parser.parse(sentence1):
    print(tree)
    tree.pretty_print()

print("________________________________________________________________________________________________________________________________________")

for tree in parser.parse(sentence2):
    print(tree)
    tree.pretty_print()

print("________________________________________________________________________________________________________________________________________")


for tree in parser.parse(sentence3):
    print(tree)
    tree.pretty_print()