import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
from typing import Tuple

file_path = "nlp_input.txt"
with open(file_path) as fh:
    text = fh.read()

    # tokenize words
    print("WORD TOKENS:")
    word_tokens = nltk.word_tokenize(text)
    print(str(word_tokens[:5]) + " ... and {} others.".format(len(word_tokens[5:])))
    print()

    # Lemmatization
    print("LEMMATIZATION:")
    lemmatizer = WordNetLemmatizer()
    lems = []
    for word in word_tokens:
        lem = lemmatizer.lemmatize(word)
        if lem != word:
            lems.append(lem)
            if 0 < len(lems) <= 5:
                print(word + " => " + lem)

    print("ect...")
    print()

    # Trigrams
    print("TRIGRAMS:")
    trigrams = list(nltk.trigrams(word_tokens))
    print(str(trigrams[:5]) + " ... and {} others.".format(len(trigrams[5:])))
    print()

    # Most common Trigrams
    freq = Counter(trigrams)
    place = 1
    most_common_trigrams = []
    for tri, cnt in freq.most_common(10):
        print(place, ": ", " Count:", cnt, " - ", tri)
        place += 1
        most_common_trigrams.append(tri)
    print()

    # Extract sentences with most common trigrams
    def is_tri_in_sent(tri_tup: Tuple[str, str, str], sent: str) -> bool:
        return (tri_tup[0] in sent) and (tri_tup[1] in sent) and (tri_tup[2] in sent)


    # tokenize sentences
    print("SENTENCE TOKENS:")
    sentence_tokens = nltk.sent_tokenize(text)
    print(str(sentence_tokens[:5]) + " ... " + " ... and {} others.".format(len(sentence_tokens[5:])))
    print()

    result = ""
    for sentence in sentence_tokens:
        for trigram in most_common_trigrams:
            if is_tri_in_sent(trigram, sentence):
                result += sentence.strip() + " "
                break
    print(result)
