from sqlite3 import dbapi2 as sqlite
import re
import math


def getwords(doc):
	splitter = re.compile('\\W*')
	# Split the words by non-alpha characters
	words = [s.lower() for s in splitter.split(doc)
				if len(s)>2 and len (s)<20]

	# Return the unique set of words only
	return dict([(w,1) for w in words])

class classifier:
	def __init__ (self,getfeatures,filename =None):
		#Counts of feature/category combinations
		self.fc= {}
		#Counts of documents in each category
		self.cc ={}
		self.getfeatures = getfeatures
		# Here getfeatures would be the getwords function

	def setdb(self,dbfile):
		self.con = sqlite.connect(dbfile)
		self.con.execute('create table if not exists fc(feature,category,count)')
		self.con.execute('create table if not exists cc(category,count)')


	#Increase the count of a feature/category pair
	def incf(self,f,cat):
		#self.fc.setdefault(f,{})
		#self.fc[f].setdefault(cat,0)
		#self.fc[f][cat]+=1
		count = self.fcount(f,cat)
		if count == 0:
			self.con.execute("insert into fc values ('%s','%s',1)" % (f,cat))
		else:
			self.con.execute("update fc set count= %d where feature='%s' and category='%s' "% (count+1,f,cat))

	#Increase the count of a category
	def incc(self,cat):
		#self.cc.setdefault(cat,0)
		#self.cc[cat]+=1
		count = self.catcount(cat)
		if count == 0:
			self.con.execute("insert into cc values('%s',1)" %(cat))
		else:
			self.con.execute("update cc set count=%d where category='%s'"% (count+1,cat))



	# The number of times a feature has appeared in a category
	def fcount(self,f,cat):
		#if f in self.fc and cat in self.fc[f]:
		#	return float(self.fc[f][cat])
		#return 0.0
		res = self.con.execute('select count from fc where feature="%s" and category="%s"'%(f,cat)).fetchone()

		if res==None: return 0
		else: return float(res[0])

	# The number of items in a category
	def catcount(self,cat):
		#if cat in self.cc:
		#	return float(self.cc[cat])
		#return 0
		res = self.con.execute('select count from cc where category="%s"' %(cat)).fetchone()
		if res ==None : return 0
		else : return float(res[0])

	# The total number of items
	def totalcount(self):
		#return sum(self.cc.values())
		res = self.con.execute('select sum(count) from cc').fetchone();
		if res == None: return 0
		return res[0]

	# The list of all categories
	def categories(self):
		#return self.cc.keys()
		cur = self.con.execute('select category from cc')
		return [d[0] for d in cur]


	def train(self,item,cat):
		features= self.getfeatures(item)
		#Increment the count for every feature with this category 
		for f in features:
			self.incf(f,cat)

		#Increment the count for this category
		self.incc(cat)
		self.con.commit()

	# The probability of getting a word from a given category
	def fprob(self,f,cat):
		if self.catcount(cat) == 0: return 0

		return self.fcount(f,cat)/self.catcount(cat)

	# The "weighted" probablilty
	def weightedprob(self,f,cat,prf,weight=1.0,ap=0.5):
		# Calculate current probability 
		basicprob = prf(f,cat)

		#Count the number of times the feature has appeared in all categories
		totals = sum([self.fcount(f,c) for c in self.categories()])

		#Count the weighted average 
		bp = ((weight*ap)+(totals*basicprob))/(weight+totals)
		return bp

class naivebayes(classifier):

	def __init__ (self,getfeatures):
		classifier.__init__(self,getfeatures)
		self.thresholds={}

	def setthreshold(self,cat,t):
		self.thresholds[cat]=t

	def getthreshold(self,cat):
		if cat not in self.thresholds: return 1.0
		return self.thresholds[cat]

	# Calculate Pr(Document| Category)
	def docprob(self,item,cat):
		features = self.getfeatures(item)

		#Multiply the probabilities of all the features together
		p=1
		for f in features: p*=self.weightedprob(f,cat,self.fprob)
		return p

	# In order to classify documents, you really need Pr(Category|Document)
	# Pr(Category|Document) = Pr(Document|Category)*Pr(Category)/Pr(Document)
	# Pr(Document) can be ignored since it will scale all the results by the same factor

	def prob(self,item,cat):
		catprob = self.catcount(cat)/self.totalcount()
		docprob = self.docprob(item,cat)
		return docprob*catprob

	# For a new item to be classified into a particular category, its probability must be a specified amount 
	# larger than the probability for any other category. (Threshold)

	def classify(self,item,default=None):
		probs={}
		#Find the category with the highest probability

		max = 0.0
		for cat in self.categories():
			probs[cat]= self.prob(item,cat)
			if probs[cat]>max:
				max=probs[cat]
				best=cat

		# Make sure the probability exceeds threshold*nextbest
		for cat in probs:
			if cat==best: continue
			if probs[cat]*self.getthreshold(best)>probs[best]: return default
		return best

class fisherclassifier(classifier):

	def cprob(self,f,cat):
		#The frequency of this feature in this category
		clf = self.fprob(f,cat)
		if clf == 0: return 0

		#The frequency of this feature in all the categories
		freqsum = sum([self.fprob(f,c) for c in self.categories()])

		#The probability is the frequency in this category divided by the overall frequency
		p=clf/(freqsum)

		return p

	def fisherprob(self,item,cat):

		#Multipy all the probabilities together using weighted probabilites like last time
		p=1 
		features = self.getfeatures(item)
		for f in features:
			p*=(self.weightedprob(f,cat,self.cprob)) 

		#Take the natural log and multiply by -2

		fscore = -2*math.log(p)

		#Use the inverse chi2 function to get a probability
		return self.invchi2(fscore,len(features)*2)

	def invchi2(self,chi,df):
		m=chi / 2.0
		sum = term = math.exp(-m)
		for i in range(1, df//2):
			term *= m/i
			sum += term
		return min(sum,1.0)

	def __init__(self,getfeatures):
	    classifier.__init__(self,getfeatures)
	    self.minimums={}

	def setminimum(self,cat,min):
	    self.minimums[cat]=min
  
	def getminimum(self,cat):
	    if cat not in self.minimums: return 0
	    return self.minimums[cat]

	def classify(self,item,default=None):
	    # Loop through looking for the best result
	    best=default
	    max=0.0
	    for c in self.categories():

	    	p = self.fisherprob(item,c)

	    	#Make sure it exceeds its minimum
	    	if p>self.getminimum(c) and p>max:
	    		best = c
	    		max = p
	    return best


def sampletrain(cl):
	cl.train('Nobody owns the water.','good')
	cl.train('the quick rabbit jumps fences','good')
	cl.train('buy pharmaceuticals now','bad')
	cl.train('make quick money at the online casino','bad')
	cl.train('the quick brown fox jumps','good')


## to add network support 



























