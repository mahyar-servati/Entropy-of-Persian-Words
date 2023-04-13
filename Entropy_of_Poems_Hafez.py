# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 21:56:16 2023

@author: Mahyar
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Import the text data
with open("Hafez_dataset.txt", "r", encoding="utf-8") as f:
    lines= f.readlines()
words = []
text = []
for l in lines:
    if len(l) > 10:
        text.append(l)
        mesra = l.split()
        for m in mesra:
            words.append(m)
text = " ".join(text)
text = text.replace("\n", " ")

# Frequency of each words
word_freq = defaultdict(int)
for w in words:
    word_freq[w] += 1

# Word length 
wordlength = np.zeros(len(words))
for i in range(len(words)):
    wordlength[i] = len(words[i])    
# plt.hist(wordlength, bins=30)
# plt.xlabel('Word lengths')
# plt.ylabel('Word count')
# plt.show()

wordcounts = np.zeros(15)
for w in words:
    wordcounts[len(w)] += 1
fig,ax = plt.subplots(1, figsize=(8,5))
ax.bar(range(len(wordcounts)), wordcounts)
ax.set_xlabel('Word lengths')
ax.set_ylabel('Word count')
ax.set_xticks(range(0,15))
plt.show()
fig.savefig("Word_length.png", bbox_inches = 'tight', dpi=300)

# entropy of all letters
persian_letters = 'آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی'
persian_num = len(persian_letters)
lettercounts = np.zeros(persian_num)
n = 0
for i in persian_letters:
    lettercounts[n] = text.count(i)
    n += 1
letterprob = lettercounts / sum(lettercounts)
entropy = -sum(letterprob * np.log2(letterprob)) 
fig,ax = plt.subplots(1, figsize=(8,5))
ax.bar(range(persian_num),lettercounts)
ax.set_xticks(range(persian_num))
ax.set_xticklabels(persian_letters)
ax.set_xlabel('Letter')
ax.set_ylabel('Count')
ax.set_title("Entropy = %.3f"%entropy)
plt.show() 
fig.savefig("Entropy_of_letters.png", bbox_inches = 'tight', dpi=300)

# Conditional (sequence) entropy
probmat = np.zeros((persian_num, persian_num))
for i in range(len(text)-1):
    currlet = text[i]
    nextlet = text[i+1]
    if currlet in persian_letters and nextlet in persian_letters:
        probmat[persian_letters.index(currlet),persian_letters.index(nextlet)] += 1
fig,ax = plt.subplots(1,figsize=(7,7))
ax.imshow(probmat,vmax=500)
ax.set_xlabel("Next letter")
ax.set_ylabel("Current letter")
ax.set_xticks(range(persian_num))
ax.set_yticks(range(persian_num))
ax.set_xticklabels(persian_letters)
ax.set_yticklabels(persian_letters)
ax.set_title("Probability of letters after each other")
plt.show()
fig.savefig("ProbabilityـofـlettersـafterـeachOther.png", bbox_inches = 'tight', dpi=300)

condentr = np.zeros(persian_num)
for i in range(persian_num):
    probs = probmat[i,:]
    probs = probs/ sum(probs)
    condentr[i] = -sum(probs*np.log2(probs + np.finfo(float).eps))

fig,ax = plt.subplots(1, figsize=(8,4))
ax.bar(range(persian_num), condentr)
ax.set_xticks(range(persian_num))
ax.set_xticklabels(persian_letters)
ax.set_title("Conditional Letter Entropy")
plt.show()
fig.savefig("Conditional_Entropy.png", bbox_inches = 'tight', dpi=300)