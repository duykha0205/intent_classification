import pandas as pd
import random
import json


class AugVietnamese:
    def __init__(self, wordnet_path, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1):
        self.wordnet = self.get_wordnet(wordnet_path)
        self.wordnet_keys = list(map(lambda x: " " + x.strip() + " ", self.wordnet.keys()))
        self.alpha_sr = alpha_sr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.p_rd = p_rd

    @staticmethod
    def key_length(key):
        return -len(key.split(' '))

    def get_wordnet(self, path):
        with open(path, "r") as f:
            wordnet = json.load(f)
        return wordnet

    def get_only_chars(self, line):
        # clean_line = ""

        # line = line.replace("’", "")
        # line = line.replace("'", "")
        # line = line.replace("-", " ") #replace hyphens with spaces
        # line = line.replace("\t", " ")
        # line = line.replace("\n", " ")
        # line = line.lower()

        # for char in line:
        #     if char in 'qwertyuiopasdfghjklzxcvbnm ':
        #         clean_line += char
        #     else:
        #         clean_line += ' '

        # clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
        # if clean_line[0] == ' ':
        #     clean_line = clean_line[1:]
        # return clean_line
        return line

    def synonym_replacement(self, words, n):
        new_words = words.copy()
        random_word_list = words.copy()
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                # print("replaced", random_word, "with", synonym)
                num_replaced += 1
            if num_replaced >= n:  # only replace up to n words
                break

        # this is stupid but we need it, trust me
        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')

        return new_words

    def get_synonyms(self, word):
        synonyms = set()
        # with open("drive/My Drive/CODE/HSD/word_net_vi.json", "r") as f:
        #     wordnet = json.load(f)

        for key, value in self.wordnet.items():
            if key.strip() == word:
                for v in value:
                    synonyms.add(v.strip())

            if word in synonyms:
                synonyms.remove(word)
        return list(synonyms)

    def synonym_replacement_v1(self, sentence, n):
        new_sentence = sentence
        sub_wordnet = random.sample(self.wordnet_keys, 5000)
        sub_wordnet.sort(key=self.key_length)

        for _ in range(n):
            replace_word, synonyms = self.get_synonyms_v1(sentence, sub_wordnet)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_sentence = new_sentence.replace(replace_word, synonym)

        # this is stupid but we need it, trust me
        new_words = new_sentence.split(' ')
        return new_words

    def get_synonyms_v1(self, sentence, sub_wordnet):
        synonyms = set()
        replace_word = ""
        for key in sub_wordnet:
            if key in sentence:
                replace_word = key.strip()
                for value in self.wordnet.get(replace_word):
                    synonyms.add(value.strip())
                if key in synonyms:
                    synonyms.remove(key)
                break
        return replace_word, list(synonyms)

    ########################################################################
    # Random deletion
    # Randomly delete words from the sentence with probability p
    ########################################################################
    @staticmethod
    def random_deletion(words, p):
        # obviously, if there's only one word, don't delete it
        if len(words) == 1:
            return words

        # randomly delete words with probability p
        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)

        # if you end up deleting all words, just return a random word
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words) - 1)
            return [words[rand_int]]

        return new_words

    ########################################################################
    # Random swap
    # Randomly swap two words in the sentence n times
    ########################################################################

    def random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            if len(new_words) > 0:
                new_words = self.swap_word(new_words)
        return new_words

    @staticmethod
    def swap_word(new_words):
        random_idx_1 = random.randint(0, len(new_words) - 1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words) - 1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
        return new_words

    ########################################################################
    # Random insertion
    # Randomly insert n words into the sentence
    ########################################################################

    def random_insertion(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, new_words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1 and len(new_words) > 0:
            # while len(synonyms) < 1:
            random_word = new_words[random.randint(0, len(new_words) - 1)]
            synonyms = self.get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return

        if len(new_words) > 0:
            random_synonym = synonyms[0]
            random_idx = random.randint(0, len(new_words) - 1)
            new_words.insert(random_idx, random_synonym)

    def __call__(self, sentence):
        sentence = self.get_only_chars(sentence)
        words = sentence.split(' ')
        words = [word for word in words if word != '']
        num_words = len(words)

        if len(words) <= 0:
            return sentence
        methods = ["sr", "ri", "rs", "rd"]
        method = random.choice(methods)

        if method == "sr":
            n_sr = max(1, int(self.alpha_sr * num_words))
            a_words = self.synonym_replacement_v1(sentence, n_sr)
        elif method == "ri":
            n_ri = max(1, int(self.alpha_ri * num_words))
            a_words = self.random_insertion(words, n_ri)
        elif method == "rs":
            n_rs = max(1, int(self.alpha_rs * num_words))
            a_words = self.random_swap(words, n_rs)
        else:
            a_words = self.random_deletion(words, self.p_rd)

        augmented_sentences = ' '.join(a_words)

        return augmented_sentences


stop_path = "vietnamese_stopword.csv"
wordnet_path = "word_net_vi_trim.json"

aug = AugVietnamese(wordnet_path=wordnet_path)
