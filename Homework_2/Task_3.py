import stanza
from nltk import Tree

stanza.download('en')

nlp = stanza.Pipeline('en')


def build_tree(word, sent):
    dependents = [w for w in sent.words if w.head == word.id]

    if dependents:
        return Tree(f"{word.text}\n({word.xpos})",
                    [build_tree(dep, sent) for dep in dependents])
    else:
        return f"{word.text}\n({word.xpos})"


sentences = [
    "Flying planes can be dangerous.",
    "The parents of the bride and the groom were flying.",
    "The groom loves dangerous planes more than the bride."
]

if __name__ == '__main__':
    for sentence in sentences:
        doc = nlp(sentence)
        for sent in doc.sentences:
            print(f"Sentence: {sentence}")

            for word in sent.words:
                print(f'{word.text}: {word.deprel} --> {sent.words[word.head - 1].text if word.head > 0 else "ROOT"}')
            print("\n")

            root = [word for word in sent.words if word.head == 0][0]
            tree = build_tree(root, sent)
            tree.pretty_print()
            print("\n")
