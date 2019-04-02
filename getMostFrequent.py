import sys,numpy as np

# SAMPLE USAGE:
#  python3 getMostFrequent.py accepted.txt rejected.txt 100 2

def readText(fn,ngram=1,lower=True):
 docFreq={}
 termFreq={}
 curWords=set()
 ndocs=0
 for line in open(fn):
   line = line.strip()
   if line=="<--EOD-->":
     for w in curWords: docFreq[w] = docFreq.get(w,0)+1
     curWords=set()
     ndocs+=1
     continue
   
   if lower: line=line.lower()
   text = ["SOS"]*(ngram-1)+line.split()+["EOS"]*(ngram-1)
   for iw in range(ngram-1,len(text)-ngram+1):
     gram = tuple(text[iw:iw+ngram])
     #if gram==("lstm","lstm","lstm"): print("<--",line)
     #print("---",gram)
     termFreq[gram] = termFreq.get(gram,0)+1
     curWords.add(gram)
 return termFreq,docFreq,ndocs

ng=int(sys.argv[4])
accepted,acc_doc,acc_ndoc = readText(sys.argv[1],ngram=ng)
rejected,rej_doc,rej_ndoc = readText(sys.argv[2],ngram=ng)

n_accepted = sum(accepted.values())
n_rejected = sum(rejected.values())
n = n_accepted+n_rejected

llr={}
k=0

for w in accepted:
  aw = accepted.get(w,0.5)
  rw = rejected.get(w,0.5)

  idf_a = np.log(acc_ndoc*1.0/acc_doc.get(w,0.5))
  idf_r = np.log(rej_ndoc*1.0/rej_doc.get(w,0.5))

  if acc_doc[w]<7: continue

  #print(idf_a,idf_r,acc_ndoc,rej_ndoc)

  #aw = aw/idf_a
  #rw = rw/idf_r
  
  ea = n_accepted*(aw+rw)/n
  er = n_rejected*(aw+rw)/n

  #print(aw,rw,n_accepted,n_rejected); 
 
  my_llr = 2*(aw*np.log(aw/ea)+rw*np.log(rw/er))
  if aw/n_accepted>rw/n_rejected: sign=1
  else: sign=-1
  llr[w] = (sign*my_llr,aw,rw,n_accepted,n_rejected)

  #print(llr[w])
  k+=1
  #if k>1: break

s = [(k, llr[k]) for k in sorted(llr, key=llr.get, reverse=True)]
kk=int(sys.argv[3]) 
for k,v in s[:kk]:
  kt = " ".join(k)
  print(kt,v)
