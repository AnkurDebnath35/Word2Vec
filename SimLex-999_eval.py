import sys
import gzip
import numpy
import math

from collections import Counter
from operator import itemgetter
#filename="64_32_128.txt"
''' Read all the word vectors and normalize them '''
def read_word_vectors(filename):    
  word_vecs = {}
  file_object = open(filename, 'r')
  for line_num, line in enumerate(file_object):
    line = line.strip().lower()
    word = line.split()[0]
    word_vecs[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vec_val in enumerate(line.split()[1:]):
      word_vecs[word][index] = float(vec_val)      
    ''' normalize weight vector '''
    word_vecs[word] /= math.sqrt((word_vecs[word]**2).sum() + 1e-6)        

  #sys.stderr.write("\nVectors read from: "+filename+" \n")
  return word_vecs











import math
import numpy
from operator import itemgetter
from numpy.linalg import norm

EPSILON = 1e-6

def euclidean(vec1, vec2):
  diff = vec1 - vec2
  return math.sqrt(diff.dot(diff))

def cosine_sim(vec1, vec2):
  vec1 += EPSILON * numpy.ones(len(vec1))
  vec2 += EPSILON * numpy.ones(len(vec1))
  return vec1.dot(vec2)/(norm(vec1)*norm(vec2))

def assign_ranks(item_dict):
  ranked_dict = {}
  sorted_list = [(key, val) for (key, val) in sorted(item_dict.items(),
                                                     key=itemgetter(1),
                                                     reverse=True)]
  for i, (key, val) in enumerate(sorted_list):
    same_val_indices = []
    for j, (key2, val2) in enumerate(sorted_list):
      if val2 == val:
        same_val_indices.append(j+1)
    if len(same_val_indices) == 1:
      ranked_dict[key] = i+1
    else:
      ranked_dict[key] = 1.*sum(same_val_indices)/len(same_val_indices)
  return ranked_dict

def correlation(dict1, dict2):
  avg1 = 1.*sum([val for key, val in dict1.iteritems()])/len(dict1)
  avg2 = 1.*sum([val for key, val in dict2.iteritems()])/len(dict2)
  numr, den1, den2 = (0., 0., 0.)
  for val1, val2 in zip(dict1.itervalues(), dict2.itervalues()):
    numr += (val1 - avg1) * (val2 - avg2)
    den1 += (val1 - avg1) ** 2
    den2 += (val2 - avg2) ** 2
  return numr / math.sqrt(den1 * den2)

def spearmans_rho(ranked_dict1, ranked_dict2):
  assert len(ranked_dict1) == len(ranked_dict2)
  if len(ranked_dict1) == 0 or len(ranked_dict2) == 0:
    return 0.
  x_avg = 1.*sum([val for val in ranked_dict1.values()])/len(ranked_dict1)
  y_avg = 1.*sum([val for val in ranked_dict2.values()])/len(ranked_dict2)
  num, d_x, d_y = (0., 0., 0.)
  for key in ranked_dict1.keys():
    xi = ranked_dict1[key]
    yi = ranked_dict2[key]
    num += (xi-x_avg)*(yi-y_avg)
    d_x += (xi-x_avg)**2
    d_y += (yi-y_avg)**2
  return num/(math.sqrt(d_x*d_y))










import sys
import os
score_file=open("scores.txt","w")
files=["64_32_128.txt","64_64_128.txt","64_32_256.txt","64_64_256.txt","128_32_128.txt","128_64_128.txt","128_32_256.txt","128_64_256.txt"]
#files=["64_32_128.txt","64_64_128.txt","64_32_256.txt","64_64_256.txt","128_32_128.txt","128_64_128.txt","128_32_256.txt","128_64_256.txt"]
score_file.write('=================================================================================\n')
score_file.write("%6s" %"Serial")
score_file.write( "%20s" % "Dataset")
score_file.write("%15s" % "Num Pairs")
score_file.write("%15s" % "Not found")
score_file.write("%15s" % "Rho\n")

score_file.write('=================================================================================\n')
print('=================================================================================')
print("%6s" %"Serial", "%20s" % "Dataset", "%15s" % "Num Pairs", "%15s" % "Not found", "%15s" % "Rho")
print('=================================================================================')

filename="EN-SIMLEX-999.txt"
for i,word_vec_file in enumerate(files):

    if __name__=='__main__':  
        
      #word_sim_dir = "C:\Users\admin\Desktop\Test Series 2018\GATE 2018\IISc\NLU\Assignment1\data\word-sim"
     
      word_vecs = read_word_vectors(word_vec_file)
    
    
      #for i, filename in enumerate(os.listdir(word_sim_dir)):
    manual_dict, auto_dict = ({}, {})
    not_found, total_size = (0, 0)
    for line in open(filename,'r'):
      line = line.strip().lower()
      word1, word2, val = line.split()
      if word1 in word_vecs and word2 in word_vecs:
        manual_dict[(word1, word2)] = float(val)
        auto_dict[(word1, word2)] = cosine_sim(word_vecs[word1], word_vecs[word2])
      else:
        not_found += 1
      total_size += 1    
    print("%6s" % str(i+1), "%20s" % filename, "%15s" % str(total_size),"%15s" % str(not_found),"%15.4f" % spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict)))
    score_file.write("%6s" % str(i+1))
    score_file.write("%20s" % filename)
    score_file.write("%15s" % str(total_size))
    score_file.write("%15s" % str(not_found))
    score_file.write("%15.4f" % spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict)))
    score_file.write("\n")
score_file.close()



#print("%15s" % str(not_found))
#print("%15.4f" % spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict)))






