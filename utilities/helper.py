import spacy
import re
import json
import pandas as pd
import mtcnn
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
import requests
import cv2
import numpy as np
def parse_sentences(inputs):
    return inputs.split('\\n')


class Tokenizer:
    def __init__(self, vi_alias_path, vi_stopword_path):
        # vi
        self.alias = json.load(open(vi_alias_path, 'r'))
        self.vi_stopwords = self.load_stopwords(vi_stopword_path)
        self.pattern = '[a-z%s]+' % re.escape("ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ")

        # en
        self.parser = spacy.load('en_core_web_sm')
        self.delete_list = set(list(
            {'ten', 'system', 'further', 'the', 'y', 'wherever', 'mill', '.', '%', '(', 'wherein', 'at', '#', 'only',
             'didn', 'to', 'shan', 'co', '}', 'out', 'beyond', 'six', 'as', 'anyway', 'so', 'than', 'together',
             'made', 'although', 'behind', 'seemed', 'otherwise', '+', 'seeming', 'nobody', 'and', 'everyone', '^',
             'must', 'can', 'do', 'by', 'alone', 'every', 'had', 'thus', 'bottom', 'thence', 'such', 'until', 'down',
             'onto', 'therein', 'part', 'we', 'ain', 'towards', 'couldn', 'whereas', 'from', 'fire', 'hence', 'under',
             'again', 'inc', 'doing', 'seem', 'here', 'mine', 'last', 'needn', 'sixty', 'amongst', 'within', 'who',
             'whole', 'without', 'its', 'thru', 'during', 'everything', 'whose', 'on', 'very', '>', 'whereupon', 'hers',
             'ie', 'often', 'around', 'there', 'several', 'beforehand', 'before', 'should', 'amoungst', 'whereby',
             'interest', '=', "she's", 'were', 'anyone', 'however', '...', 'he', 'name', 'now', 'yourselves', 'shouldn',
             'both', 'one', 'con', 'even', '|', 'most', 'noone', 'don', 'this', 'nevertheless', 'empty', 'serious',
             'front', 'theirs', 'about', 'found', 'has', 'wasn', 'whither', '-', 'of', 'own', 'though', ':', 'another',
             'therefore', 'always', 'ma', 'becoming', 'go', 'sincere', 'hereupon', 'once', 'too', 'hereafter', 'bill',
             'move', 'up', 'just', 'neither', 'two', 'nowhere', 'see', '"', 'his', 'many', 'mightn', 'side',
             'thereupon', 'those', 'either', 'are', 've', 'formerly', 'please', 'eight', 'into', 'might', 'full',
             'rather', 'd', 'twelve', "you've", 'their', 'after', '*', 'forty', 'back', 'cant', 'in', 'am',
             'except', 'yours', 'yet', '@', 'may', 'a', 'hasn', 'below', 'him', '&', 'something', 'much', '$', 'hasnt',
             'any', 'each', 'hadn', 'besides', 'still', 'throughout', 'latterly', 'all', 'toward', 'fill', ']',
             '~', 'aren', 't', 're', '_', 'due', 'won', 'done', 'else', 'your', 'via', 'these',
             'whence', 'been', 'across', 'it', 'couldnt', "should've", 'while', 'few', 'meanwhile', 'you', 'thin',
             'least', 'whom', 'same', 'other', 'perhaps', 'whereafter', 'four', 'will', 'off', 'others', '[',
             '/', 'itself', '?', 'whenever', 'me', 'also', 'weren', 'former', 's', 'yourself', '<', 'became', 'first',
             'for', 'whether', 'above', 'less', 'un', "it's", 'us', 'have', 'being', "''", 'sometimes', 'she', 'if',
             'beside', 'afterwards', 'did', 'mustn', 'does', 'be', 'll', 'isn', ';', 'becomes', 'per', 'upon', 'haven',
             'between', 'namely', 'because', 'sometime', 'de', 'cry', 'over', 'my', 'enough', 'five', 'eleven',
             'top', 'then', "you're", "you'll", 'thick', 'nine', 'is', 'himself', 'herself', 'since', 'could', "you'd",
             'along', 'ever', 'or', 'they', 'anyhow', 'hundred', ')', 'herein', 'through', 'an',
             'thereby', ',', 'elsewhere', 'already', 'some', 'with', 'o', 'themselves', 'twenty', 'somewhere', 'among',
             'ours', 'call', '`', 'm', 'etc', 'against', '!', 'moreover', 'latter', 'anywhere', 'seems', 'someone',
             'fifteen', 'more', 'fifty', 'our', 'ltd', 'third', 'was', 'that', 'show', "'", "that'll", 'indeed',
             'three', 'everywhere', 'mostly', 'become', 'whoever', 'would', 'next', 'doesn', 'hereby', 'wouldn',
             'ourselves', 'almost', 'eg', '{', 'thereafter', 'somehow', 'whatever', 'having', 'well', 'her', '”', 'i',
             '\\', 'myself', 'them'}))

    def en_tokenize(self, text):
        text = text.strip().replace("\n", " ").replace("\r", " ")
        text = text.lower()  # lower case
        tokens = self.parser(text)
        # lemmatization
        lemmas = []
        for tok in tokens:
            lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
        tokens = lemmas

        tokens = [tok for tok in tokens if tok.lower() not in self.delete_list]
        # remove remaining tokens that are not alphabetic
        tokens = [tok for tok in tokens if tok.isalpha()]

        return ' '.join(tokens[:])

    def vi_tokenize(self, text):
        # extract words
        tokens = re.findall(self.pattern, str(text).lower())
        text = ' %s ' % ' '.join(tokens)
        # replace aliases
        for word in self.alias.keys():
            text = text.replace(' ' + word + ' ', ' ' + str(self.alias[word]) + ' ')
        # remove stopwords
        for word in self.vi_stopwords['word'].values:
            text = text.replace(' ' + word + ' ', ' ')
        return text.strip()

    @staticmethod
    def load_stopwords(path):
        stopwords = pd.read_csv(path)
        stopwords = stopwords[stopwords['label'] == 0]

        return stopwords


class FaceHelper():
    def __init__(self, tf_serving_url):
        self.__detector = mtcnn.MTCNN()
        self.__extractor_url = tf_serving_url # http://127.0.0.1:8501/v1/models/face_emb:predict


    def extract(self, img_list):
        image = np.reshape(img_list,(len(img_list),112,112,3))
        x = {"signature_name": "serving_default", "instances": image.tolist()}

        headers = {"content-type": "application/json"}

        data = json.dumps(x)

        resp = requests.post(self.__extractor_url, data=data, headers=headers)
        if resp.status_code == 200:
            predictions = json.loads(resp.text)['predictions']
            predictions = preprocessing.normalize(predictions, norm="l2")
            return predictions
        else:
            print('Cannot call to serving ', resp.status_code)
            return None


    def extract_feature(self, image):
        margin = 5
        h,w,_ = image.shape
        faces = self.__detector.detect_faces(image)
        for face in faces:
            # Assume that face detector is very accurate
            x,y,width,height = face["box"]

            if y-margin < 0:
                y1 = 0
            else:
                y1 = y - margin

            if x-margin < 0:
                x1 = 0
            else:
                x1 = x-margin

            if x+width+margin > w:
                x2 = w
            else:
                x2 = x+width+margin

            if y+height+margin > h:
                y2 = h
            else:
                y2 = y+height+margin


            face_image1 = image[y1:y2,x1:x2]
            break

        face1 = cv2.resize(face_image1, (112,112))
        result1 = np.array(self.extract([face1])[0])

        return result1


    def match(self, image1, image2):
        margin = 5

        # Process image
        h,w,_ = image1.shape
        faces = self.__detector.detect_faces(image1)
        for face in faces:
            # Assume that face detector is very accurate
            x,y,width,height = face["box"]

            if y-margin < 0:
                y1 = 0
            else:
                y1 = y - margin

            if x-margin < 0:
                x1 = 0
            else:
                x1 = x-margin

            if x+width+margin > w:
                x2 = w
            else:
                x2 = x+width+margin

            if y+height+margin > h:
                y2 = h
            else:
                y2 = y+height+margin


            face_image1 = image1[y1:y2,x1:x2]
            break

        face1 = cv2.resize(face_image1, (112,112))
        result1 = np.array(self.extract([face1])[0])

        # Process image
        h,w,_ = image2.shape
        faces = self.__detector.detect_faces(image2)
        for face in faces:
            # Assume that face detector is very accurate
            x,y,width,height = face["box"]

            if y-margin < 0:
                y1 = 0
            else:
                y1 = y - margin

            if x-margin < 0:
                x1 = 0
            else:
                x1 = x-margin

            if x+width+margin > w:
                x2 = w
            else:
                x2 = x+width+margin

            if y+height+margin > h:
                y2 = h
            else:
                y2 = y+height+margin


            face_image2 = image2[y1:y2,x1:x2]
            break

        face2 = cv2.resize(face_image2, (112,112))
        result2 = np.array(self.extract([face2])[0])

        dist = euclidean_distances([result1],[result2])[0]

        return dist

tokenize = Tokenizer(vi_alias_path="utilities/alias.json", vi_stopword_path="utilities/vietnamese_stopword.csv")
face_helper = FaceHelper("http://192.168.100.2:8501/v1/models/face_emb:predict") # Temporary hard code
