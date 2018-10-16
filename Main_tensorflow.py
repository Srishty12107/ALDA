import os
import sys

import time
import math
import random
import zipfile
import collections 

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf



for s in sys.argv:
    print(s)

num_steps = int(sys.argv[1])

#print("importing nltk")
#import nltk

#from nltk.tag import StanfordPOSTagger
#st = StanfordPOSTagger('english-bidirectional-distsim.tagger')

#strng = "Using a part of speech tagger from main in python"
#print(strng)
#print("nltk.pos_tag:")
#print([s[1] for s in nltk.pos_tag(strng.split())], "\n")
#print("st.tag:")
#print(st.tag(strng.split()), "\n")
#time.sleep(2)
#os.system("./f")



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
    #return ""




url = 'http://mattmahoney.net/dc/'
def maybe_dl(f, e_b):
    if not os.path.exists(f):
        f, _ = urllib.request.urlretrieve(url + f, f)
        #print 'requesting',f,'from',url
    statinfo = os.stat(f)
    if statinfo.st_size == e_b:
        #print 'have file',f
        pass
    else:
        print statinfo.st_size
        raise Exception('Can\'t verify ' + f) 
    return f

#filename = maybe_dl('text8.zip', 31344016)
filename = "" #maybe_dl('Newtest48.txt', 1370184)







def read_data(fn):
    #with zipfile.ZipFile(fn) as f:
    #data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    
    data = []
    for i in range(1):
        #f = fn + "test" + str(i) + ".txt"
        #f = "second_test.txt"
        f = "allcfr_test.txt"
        if os.path.exists(f):
            da = open(f,"r")
            #data += [ rmAlph(t).lower() for t in tf.compat.as_str(da.read()).split() if len(rmAlph(t)) > 1]
            data += [ t.lower() for t in tf.compat.as_str(rmAlph(da.read())).split() if 'http' not in t]
            da.close()
    
    #for d in data:
    #    print d
    return data

vocabulary = read_data(filename)
print 'vocab size is',len(vocabulary)
#print 'type of vocab is',type(vocabulary)
vocabulary_size = 50000









def build_dataset(wrds, n_wrds):
    count = [['unk',1]]
    count.extend(collections.Counter(wrds).most_common(n_wrds-1))
    dictionary = dict()
    for wrd, _ in count:
        dictionary[wrd] = len(dictionary)
    data = list()
    unk_count = 0
    for wrd in wrds:
        if wrd in dictionary:
            index = dictionary[wrd]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

data,count,dictionary,reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
del vocabulary







data_index = 0
vocabulary_size = len(dictionary)

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    lables = np.ndarray(shape=(batch_size,1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0,span-1)
            targets_to_avoid.append(target)
            batch[i*num_skips+j] = buffer[skip_window]
            lables[i*num_skips+j] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
        data_index = (data_index + len(data) - span) % len(data)
    return batch, lables
                
batch, lables = generate_batch(batch_size=128,num_skips=2,skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', lables[i,0], reverse_dictionary[lables[i,0]])


















batch_size, embedding_size, skip_window, num_skips = 128, 128, 1, 2
valid_size, valid_window, num_sampled = 75, 100, 64
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_lables = tf.placeholder(tf.int32, shape =[batch_size,1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, 
                                                       embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
    loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                         biases = nce_biases,
                                         labels = train_lables,
                                         inputs = embed,
                                         num_sampled = num_sampled,
                                         num_classes = vocabulary_size))
    
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    
#num_steps = 750001

outfile = open("all_cfr_results2_"+str(num_steps)+".txt","w")

with tf.Session(graph=graph) as session:
    init = tf.global_variables_initializer()
    init.run()
    #print 'initialized'
    #outfile.write('initialized\n')
    average_loss = 0
    
    for step in xrange(num_steps):
        batch_inputs, batch_lables = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs : batch_inputs, train_lables:batch_lables}
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print 'averge loss at step ',step,' is',average_loss
            outfile.write('averge loss at step '+str(step) + ' is ' + str(average_loss) +"\n")
            average_loss = 0
            
        if step % 10000 == 0:
            sim = similarity.eval()
            
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 24
                nearest = (-sim[i,:]).argsort()[1:top_k+1]
                log_str = 'nearest to %s: ' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s' % (log_str, close_word)
                #print log_str
                outfile.write(log_str+"\n")
                    
    final_embeddings = normalized_embeddings.eval()

outfile.close()
print "finished writing f.py with num_steps =",num_steps


    



def main():
    pass



#end main


'''
https://www.tensorflow.org/tutorials/word2vec
https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
https://lvdmaaten.github.io/tsne/
http://www.cs.toronto.edu/~hinton/absps/tsnefinal.pdf

http://scikit-learn.org/stable/modules/manifold.html
http://lvdmaaten.github.io/tsne/
https://www.cs.cmu.edu/~efros/courses/AP06/presentations/ThompsonDimensionalityReduction.pdf
http://www.lcayton.com/resexam.pdf


'''








main()


# verizon opt out
# http://privacy.aol.com/mobile-choices/



# tor
# https://libraryfreedomproject.org/torexitpilotphase1/
# https://tor.stackexchange.com/questions/1435/connecting-to-tor-through-verizon#1444

# tf
# http://scikit-learn.org/stable/modules/manifold.html
# https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py
# https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec_optimized.py
# https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

# to dl:
# https://nlp.stanford.edu/software/
# https://help.ubuntu.com/community/Java


# https://www.csee.umbc.edu/~jtang/cs421.s17/homework/hw2.html
# https://www.csee.umbc.edu/courses/undergraduate/CMSC473/
# https://www.youtube.com/watch?v=QvQAl-zbiCI
# https://www.tensorflow.org/install/install_linux#installing_with_native_pip


# https://www.mongodb.com/
# http://www.tutorialspoint.com/mongodb/
# https://docs.mongodb.com/
# https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/






# https://neo4j.com/download/
# http://www.ngrams.info/download_coca.asp
# https://duckduckhack.com/
# https://developer.uspto.gov/product/patent-grant-dataxml
# https://www.uspto.gov/learning-and-resources/ip-policy/economic-research/research-datasets
# https://www.uspto.gov/learning-and-resources/open-data-and-mobility
# http://appft1.uspto.gov/netacgi/nph-Parser?Sect1=PTO1&Sect2=HITOFF&d=PG01&p=1&u=/netahtml/PTO/srchnum.html&r=1&f=G&l=50&s1=20140075004.PGNR.
# http://www.cs.cmu.edu/~dpelleg/download/
# 



for s in sys.argv:
    print(s)
# source /home/ebig/bin/activate
