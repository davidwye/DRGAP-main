import difflib
import re
def diff_word(sentence1, sentence2):
    words1 = sentence1.lower().split()
    words2 = sentence2.lower().split()

    differ = difflib.Differ()
    diff = list(differ.compare(words1, words2))

    res = []
    for line in diff:
        if line.startswith('- '):
            res.append(re.sub(r'[^A-Za-z-`]', '', line[2:]))
        if line.startswith('+ '):
            res.append(re.sub(r'[^A-Za-z-`]', '', line[2:]))
    return res[0],res[1]
