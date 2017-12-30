from sklearn.feature_extraction.text import TfidfVectorizer
import textbook
import pandas
import preprocessing

"""
Computes the similarity between essays and each chapter of the textbook
We adpoted TFIDF as the similarity measure
1. find out 50 words from each chapter with the most occurence, so it would be 15*50 in total
2. use the 750 words to find out the TFIDF values of the chapters
3. for each essay, count the words to find the TF value 
	and multiply with the IDF values from step 2 to obtain TFIDF of essays
4. compute the pairwise similarity between chapters and essays by taking dot products
"""

sample_folder = "./samples"
out_file = "similarity.csv"
n_key_vocabs = 50
ngram_rng = (1, 3)

chs = ['1a','1b','2','3a','3b','4','5','6','7','8','9','10a','10b','11a','11b']

samples = preprocessing.tp_sample.get_samples(sample_folder)
samples_textbook = [sample.text for sample in samples]+textbook.getOrderedText(chs = chs)
vocabularies = {}
for ch in chs:
	ch_vocabs = textbook.getTopVocabs(ch, n = n_key_vocabs, ngram_rng = ngram_rng)
	for vocab, freq in ch_vocabs:
		vocabularies[vocab] = 1
vectorizer = TfidfVectorizer(ngram_range = ngram_rng,  stop_words = 'english', vocabulary = vocabularies.keys())
vectorizer.fit(textbook.getOrderedText(chs = chs))

tfidf = vectorizer.transform(samples_textbook)

similarity = (tfidf*tfidf.T).A
similarity = similarity[0:len(samples), len(samples):]

similarity_df = pandas.DataFrame(similarity, columns = textbook.getChapterTitles())
similarity_df.index = [sample.get_identifier() for sample in samples]

similarity_df.to_csv(out_file)
