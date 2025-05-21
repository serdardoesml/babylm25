# prep_gutenberg.py

import os
import sys
import sentence_splitter
from tqdm import tqdm

def clean_gutenberg(path):
    new_lines = []
    starts = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            if line.upper() == line: # header
                continue
            elif "_________" in line: # blank line
                continue
            elif "= = =" in line: # metadata line
                continue
            elif line[0] == "[" and line.strip()[-1] == "]": # footnote or illustration
                continue

            for sent in sentence_splitter.split_text_into_sentences(line, language="en"):
                start = " ".join(sent.split(" ")[:4])
                if start in starts: # duplicate start
                    continue
                new_lines.append(sent)
                starts[start] = True

    return new_lines

def clean_simple_wiki(path):
    new_lines = []
    starts = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            if "= = =" in line: # title line
                continue
            elif len(line.split(" ")) < 4: # most subtitle lines
                continue
            
            for sent in sentence_splitter.split_text_into_sentences(line, language="en"):
                sent = sent.replace("&amp", "##AMP##") # mark legit &s                
                if "&" in sent: # remove other markers
                    continue
                sent = sent.replace("##AMP##", "&") # convert back
            
                start = " ".join(sent.split(" ")[:4])
                if start in starts: # duplicate start
                    continue
                new_lines.append(sent)
                starts[start] = True
    return new_lines

if __name__ == "__main__":
    path = sys.argv[1]
    new_path = sys.argv[2]

    if "gutenberg" in path:
        new_lines = clean_gutenberg(path)
    elif "simple_wiki" in path:
        new_lines = clean_simple_wiki(path)
    else:
        raise NotImplementedError

    with open(new_path, "w") as f:
        f.write("\n".join(new_lines))
