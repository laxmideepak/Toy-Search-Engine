import os
import math
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

corpusroot = ''
docContent=dict()
tf= dict()
wtf=dict()
idf = dict()
tfidf = dict()
cosinemagnitude = dict()
N = 30

def tokenizeAndStemming(docValue):
    docValue = docValue.lower()
    # Tokenize the document using Regex expression
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokensList = tokenizer.tokenize(docValue)                 
        
    # Stopwords List
    stopwordsList= stopwords.words('english')

    # Using Porter Stemmer algorithm
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokensList if token not in stopwordsList]
    return stemmed_tokens

def calculateValues(docToken):
    tfcount=dict()
    docno=0
    for sentence in docToken.values():
        for stemmedword in sentence:
            if stemmedword not in tfcount:
                tfcount.setdefault(stemmedword, [0] * N)
                tfcount[stemmedword][docno] += 1
            else:
                tfcount[stemmedword][docno] += 1
        docno += 1

    weightedtf = dict()
    for stemmedword in tfcount:
        weightedtf.setdefault(stemmedword, [0] * N)
        for col in range(0, N):
            if tfcount[stemmedword][col] == 0:
                weightedtf[stemmedword][col] = 0
            else:
                weightedtf[stemmedword][col] = 1 + (math.log10(tfcount[stemmedword][col]))

    inversedf = dict()
    for stemmedword in tfcount:
        inversedf.setdefault(stemmedword,N)
        df = 0
        for col in range(0, N):
            if tfcount[stemmedword][col] > 0:
                df += 1
        final_value=N/df
        inversedf[stemmedword] = math.log10(final_value)   

    wtfIDF = dict()
    for stemmedword in weightedtf:
        wtfIDF.setdefault(stemmedword, [0] * N)
        for col in range(0, N):
            if weightedtf[stemmedword][col] == 0:
                wtfIDF[stemmedword][col] = 0
            else:
                wtfIDF[stemmedword][col] = weightedtf[stemmedword][col] * inversedf[stemmedword]

    magnitude = dict()
    for doccount in range (0,N):
        square =0 
        keys_list = list(docContent.keys())
        magnitude.setdefault(keys_list[doccount],N)
        for stemmedword in wtfIDF:
             square += wtfIDF[stemmedword][doccount] ** 2
        magnitude[keys_list[doccount]] = math.sqrt(square)

    return tfcount , weightedtf , inversedf , wtfIDF , magnitude

def getidf(word):
    stemmedword = tokenizeAndStemming(word)
    if stemmedword[0] in idf:
        return idf[stemmedword[0]]
    return -1

def getweight(userfilename, word):
    if userfilename in docContent:
        keys_list = list(docContent.keys())
        index = keys_list.index(userfilename)    
        stemmedword = tokenizeAndStemming(word)
        if stemmedword[0] in tfidf:
            score = tfidf[stemmedword[0]][index]
            normalizedscore= score / cosinemagnitude [userfilename]
            return normalizedscore
    return 0

def query(word):
    stemmedquerylist = tokenizeAndStemming(word)
    queryscore = dict()
    maxsimilarity = dict()
    querysimilarity = dict()

    queryoccurence=Counter(stemmedquerylist)
    for term , counts in queryoccurence.items():
        if counts == 0 :
            queryscore[term] = 0
        else:
            queryscore[term] = 1 + (math.log10(counts)) 
    squares = sum(x**2 for x in queryscore.values())
    magnitude = math.sqrt(squares)
    for key, value in queryscore.items():
        querysimilarity[key] = value / magnitude

    for filename in docContent.keys():
        keys_list = list(docContent.keys())
        index = keys_list.index(filename)
        cosinedotproduct = sum(queryscore[term] * tfidf[term][index] for term in stemmedquerylist if term in tfidf)
        docmagnitude = cosinemagnitude[filename]
        if docmagnitude != 0 and magnitude != 0:
            cosinesimilarityscore = cosinedotproduct / (docmagnitude * magnitude)
            maxsimilarity[filename] = cosinesimilarityscore
    
    max_key = max(maxsimilarity, key=lambda k: maxsimilarity[k])
    max_value = maxsimilarity[max_key]
    
    return max_key , max_value

for filename in os.listdir(corpusroot):
    if filename.startswith('0') or filename.startswith('1') or filename.startswith('2') or filename.startswith('3'):
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        doc = file.read()
        file.close()        
        docContent[filename] = tokenizeAndStemming(doc)

tf, wtf , idf , tfidf , cosinemagnitude = calculateValues(docContent)

print("%.12f" % getidf('children'))
print("%.12f" % getidf('foreign'))
print("%.12f" % getidf('people'))
print("%.12f" % getidf('honor'))
print("%.12f" % getidf('great'))
print("%.12f" % getidf('chandramouli'))
print("--------------")
print("%.12f" % getweight('19_lincoln_1861.txt','constitution'))
print("%.12f" % getweight('23_hayes_1877.txt','public'))
print("%.12f" % getweight('25_cleveland_1885.txt','citizen'))
print("%.12f" % getweight('09_monroe_1821.txt','revenue'))
print("%.12f" % getweight('05_jefferson_1805.txt','press'))
print("%.12f" % getweight('05_jefferson_1805.txt','chandramouli'))
print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("war offenses"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("texas government"))
print("(%s, %.12f)" % query("cuba government"))
print("(%s, %.12f)" % query("chandramouli"))
