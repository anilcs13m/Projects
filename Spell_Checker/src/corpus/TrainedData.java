package corpus;

public class TrainedData {
	
	Words unigramData;
	Trigram trigramData;
	Bigram bigramData;
	

	public TrainedData() {
		unigramData = new Words();
		trigramData = new Trigram();
		bigramData = new Bigram();
		
	}
	
	public int count(String ngram) {
		
		String [] words=ngram.split(" ");
		if(words.length==3)		
			return trigramData.count(ngram);
		
		if(words.length==2)
			return bigramData.count(ngram);
		else
			return unigramData.count(ngram);
	}
	
	public double trigramPrior(String word, String history) {
		return trigramData.prior(word,history);
	}
	public double bigramPrior(String word, String history) {
		return bigramData.prior(word,history);
	}
	public double unigramPrior(String word) {
		return unigramData.prior(word);
	}
	
	public boolean hasWord(String word){
		if(unigramData.count(word)>0){
			return true;
		} else
			return false;
	}
		
}
	

