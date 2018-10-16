import os
import sys
import zipfile
import html5lib
from math import log 
from lxml import etree
from cStringIO import StringIO
from nltk.corpus import stopwords

STOP_WORDS = list(set(stopwords.words('english')))

def rmAlph(text):
    C = ""
    if isinstance(text,str):
        for c in text:
            t = ord(c)
            if t > 96 and t < 123:
                C += c
            elif t > 64 and t < 91:
                C += c
            elif c == ' ':
                C += ' '
            elif t == 10:
                C += ' '
            elif t == 9:
                C += ' '
    #if 'http' not in C:        
    return C


QUERY_WORDS = []
if os.path.exists("query_words.txt"):
    infile = open("query_words.txt","r")
    QUERY_WORDS = [ rmAlph(l.lower()) for l in infile.read().split('\n') if len(l) > 1 ]
    infile.close()

#for qw in QUERY_WORDS:
#    print qw
#print "len qw =",len(QUERY_WORDS)






def call(ret, phrase,lst):
    if len(lst) == 0:
        if len(phrase) > 1:
            p = phrase.lower()
            if p not in STOP_WORDS:
                ret.append(p)
    elif len(lst) == 1:
        for p in phrase.split(lst[0]): call(ret, p, "")
    else:
        for p in phrase.split(lst[0]): call(ret, p, lst[1:])


    
def cln_strng(text):
    retrn = []
    if isinstance(text, str):
        strp = " "
        for t in text:
            o = ord(t)
            if o < 65 or o > 122:
                if t not in strp:
                    strp = t + strp
            elif o < 97 and o > 90:
                if t not in strp:
                    strp = t + strp
        if len(strp) > 1:
            for c in text.split(strp[0]): call(retrn, c, strp[1:])
        else:
            for c in text.split(strp[0]): call(retrn, c, "")
    return retrn




















class FileHandler():
    def __init__(self, zfile):
        self.zfile = zfile

    def readline(self):
        return self.zfile.readline()

    def listXmls(self):
        output = StringIO()
        line = self.readline()
        output.write(line)
        line = self.readline()
        while line is not '':
            if '<?xml version="1.0" encoding="UTF-8"?>' in line:
                line = line.replace('<?xml version="1.0" encoding="UTF-8"?>', '')
                output.write(line)
                output.seek(0)
                yield output
                output = StringIO()
                output.write('<?xml version="1.0" encoding="UTF-8"?>')
            elif '<?xml version="1.0"?>' in line:
                line = line.replace('<?xml version="1.0"?>', '')
                output.write(line)
                output.seek(0)
                yield output
                output = StringIO()
                output.write('<?xml version="1.0"?>')
            else:
                output.write(line)
            try:
                line = self.readline()
            except StopIteration:
                break
        output.seek(0)
        yield output


class SXMLH(object):
    def __init__(self):
        self.CONTENTS = 0
        self.SUBPART = 0
        self.SECTION = 0
        self.SECTNO = 0
        self.SUBJECT = 0
        self.P = 0
        self.bad_subjs = ["Scope of subpart.","Scope of part.","Scope."]
        self.new_sect = 0
        
        self.SUBPART_text = ''
        self.SECTION_text = ''
        self.SECTNO_text = ''
        self.SUBJECT_text = ''
        self.P_text = ''
        
        self.buffer = ''
        self.write_buffer = []

    def start(self, tag, attributes):
        if tag == 'CONTENTS':
            self.CONTENTS = 1
        elif tag == 'SUBPART':
            self.SUBPART = 1
        elif tag == 'SECTION':
            self.SECTION = 1
        elif tag == 'SECTNO':
            self.SECTNO = 1
            if self.CONTENTS == 0:
                if len(self.P_text) > 0:
                    if len(self.SUBJECT_text) > 0:
                        self.write_buffer.append((self.SECTNO_text, self.SUBJECT_text, self.P_text))
        elif tag == 'SUBJECT':
            self.SUBJECT = 1
        elif tag == 'P':
            self.P = 1
        else:
            pass
        #linear or quadratic gaussian economics
        #linear or quadratic reaction function
        # glide pass hockey stick jump conditions
    def data(self, data):
        if self.CONTENTS == 0:
            if self.SECTION == 1:
                if self.SECTNO == 1:
                    self.SECTNO_text = data
                elif self.SUBJECT == 1:
                    if data.encode('utf-8') not in self.bad_subjs:    
                        self.SUBJECT_text = data
                    else:
                        self.SUBJECT_text = ''
                elif self.P == 1:
                    #if len(self.P_text) == 0:
                    self.P_text += (str(data.encode('utf-8').rstrip())+'\n')
                    #else:
                    #    self.P_text += '\n'
                    #    self.P_text += data
                else:
                    pass
            else:
                pass

    def end(self, tag):
        if tag == 'CONTENTS':
            self.CONTENTS = 0
        elif tag == 'SUBPART':
            self.SUBPART = 0
        elif tag == 'SECTION':
            if self.CONTENTS == 0:
                self.SECTION = 0
                #self.SECTNO_text = ''
                #self.SUBJECT_text = ''
                #self.P_text = ''
        elif tag == 'SECTNO':
            self.SECTNO = 0
            #self.SECTNO_text = ''
            self.P_text = ""
        elif tag == 'SUBJECT':
            self.SUBJECT = 0
            #self.SUBJECT_text = ''
        elif tag == 'P':
            self.P = 0
            #if self.CONTENTS == 0:
                #if self.new_para = 1:
                    #self.write_buffer.append((self.SECTNO_text, self.SUBJECT_text, self.P_text))
                
            
            
            #if self.CONTENTS == 0:
            #    self.P_text = ''
        
        else:
            pass
        
    def close(self):
        return self.write_buffer




# https://www.gpo.gov/fdsys/bulkdata/CFR/2016/title-48/CFR-2016-title48-vol1.xml
# https://github.com/hopped/uspto-patents-parsing-tools/blob/master/uspto-xml-parser.py
# 
# 
# 




# cfr tags we'll need:
#
# <TITLE> <CHAPTER> <SUBCHAP> <PART> <SUBPART> <SECTION>
# <EAR> is part number, <HD> is part description
# <SECTION> has <SECTNO>, <SUBJECT>, <P> tag(s) 
# 
# 




class CFRPart():
    def __init__(self,listv):
        self.section = listv[0].encode('utf-8')
        self.subject = listv[1].encode('utf-8') 
        self.vocab = { 'unk' : 0 } # is unique word & its count
        self.wordcount = 1
        self.text = listv[2]#.encode('utf-8')
        self.name =  str(self.section) + "  " + str(self.subject)
    def add_word(self, w):
        #w = w2.encode('utf-8')
        if w not in self.vocab:
            self.vocab[w] = 1
        else:
            self.vocab[w] += 1
        self.wordcount += 1
    #def add_text(self, strng):
    #    self.text = strng
    def get_tf(self, t):
        if t in self.vocab:
            return 1.0*self.vocab[t] / self.wordcount
        else:
            return 0.0

#cfr_title48_part1.txt
#01234567890123456789
#00000000001111111111



class CFRCorpus():
    def __init__(self):
        self.listofparts, self.vocab = [], { 'unk' : 0 }
        self.partcount = 0
    def add_part(self, p):
        self.listofparts.append(p)
        self.partcount += 1        
    def add_part_to_vocab(self, ws):
        for w in ws:
            if w in self.vocab:
                self.vocab[w] += 1
            else:
                self.vocab[w] = 1
    def get_idf(self, t):
        if t in self.vocab:
            return self.vocab[t]
        else:
            return 0
    def dump(self):
        print "Partcount: ",self.partcount
        print "Vocab len: ",len(self.vocab)


dirpath = '/home/mike/School/ebiq/cfrParser/'
#print "directory path is ", dirpath    
myCorpus = CFRCorpus() 

if os.path.isdir(dirpath):
    for zip_file in os.listdir(dirpath):
        zip_file = dirpath + zip_file
        if not zip_file.endswith('CFR-2016-title48.zip'):
            continue
        try:
            zfile = zipfile.ZipFile(zip_file, 'r')
        except zipfile.BadZipFile:
            print "Bad Zipfile."
            continue
        
        for name in zfile.namelist():
            #print "name =", name    
            if '2016' in name and name.endswith('.xml'):
                f = FileHandler(zfile.open(name, 'rU'))
                print "Opened", name
                for elem in f.listXmls():
                    parser = etree.XMLParser(target=SXMLH(), resolve_entities=False,load_dtd=False)
                    result = etree.parse(elem,parser)
                    if len(result) > 1:
                        for res in result:
                            #print res[0], "  ",res[1]
                            myPart = CFRPart(res)
                            cln = cln_strng(myPart.text.lower())
                            for c in cln: 
                                myPart.add_word(c)
                            myCorpus.add_part(myPart)
                            #print "Added:",myPart.name
            
                    


#                    for line in infile.read().split('\n'):
#                        myPart = CFRPart(name, sentence)
                            #print "Reading part no.", myPart.name
                    #line = ""
                    #myPart.add_text(line)
                    
            else:
                print "Skipping", name
            
        for m in myCorpus.listofparts:
        #    print m.filename
            myCorpus.add_part_to_vocab(list(set(m.vocab.keys())))


myCorpus.dump()

def get_tfidf(query):    
    qry = cln_strng(query)
    rets = [ [] for q in qry]
    #print rets
    doc_names = []
    idf_numer = myCorpus.partcount
    indx = 0
    for q in qry:
        #print q
        #get num docs with term
        idf_denom = myCorpus.get_idf(q.lower())+1
        idf = log(1.0 * idf_numer / idf_denom, 2)#use 2,10,e
    
        for d in myCorpus.listofparts:
            #print d.filename,
            tf = d.get_tf(q)#+1
            #print tf, idf
            rets[indx].append( 1.0*tf*idf)
            doc_names.append((d.name, d.text))
        
        indx += 1
        
    ret = [  ]
    for i in range(len(myCorpus.listofparts)):
        tot = 0.0 
        for ii in range(len(qry)):
            tot += rets[ii][i]
        ret.append( (tot, doc_names[i]) )
    #for i in range(len(myCorpus.listofparts)): ret.append(sum(rets[:][i]))
    #r = zip(ret, doc_names)
    return sorted(ret, key = lambda k: k[0], reverse=True)

QUERY_LIST = ["How does a vendor appeal an award decision?",
    "How many days at a minimum must a Request for Proposal (RFP) be posted (open, available)?",
    "What is the maximum number of days that an RFP may stay posted, open or available?",
    "What are the responsibilities assigned to a contracting officer?" ]


#ql=0

for ql in range(len(QUERY_LIST)):
    sortedDocs = get_tfidf(QUERY_LIST[ql])

    print "\n\n\n***********************************************************"
    print QUERY_LIST[ql]
    for i in range(len(sortedDocs)/100):
        #print "----------------------"
        #print("%6.4f   %s -------------------\n%s" % (sortedDocs[i][0], sortedDocs[i][1][0], sortedDocs[i][1][1][0:]))
        if sortedDocs[i][0] > 0.1:
            print("%6.4f   %s  " % (sortedDocs[i][0], sortedDocs[i][1][0]))


#for sd in sortedDocs:
 #   print sd[0], sd[1]















