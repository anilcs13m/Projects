package spellcheck;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.Map.Entry;

import corpus.TrainedData;

public class WordCheck {
	private static final int MAX_LOOPING = 220000;
	/**
	 * For correct Words ignoring context
	 */
	ConfusionMatrix cMatrix;
	private TrainedData trainedData;
	
	final static int MAX_EDIT = 3;
	final static int NO_OF_SUGGESTION = 10;

	public WordCheck(TrainedData trainedData) {
		this.cMatrix = new ConfusionMatrix();
		this.trainedData = trainedData;
	}

	/**
	 * Generate words within a single edit distance
	 * @param word - word to be correct
	 * @param allWords - all word within a single edit distance (for next recursive call if needed)
	 * @return all valid word within a single edit distance
	 */
	private Set<String> edits(final String word,
			final List<String> allWords) {

		//parallel each For

		final Set<String> validWords = Collections.synchronizedSet(new HashSet<String>());

		Thread del,rev,ins,sub;

		del = (new Thread() {
			public void run() {
				delete(word, validWords,allWords);
			}
		});
		rev = new Thread() {
			public void run() {
				reverse(word, validWords,allWords);
			}
		};
		ins = new Thread() {
			public void run() {
				insert(word, validWords,allWords);
			}
		};
		sub = new Thread() {
			public void run() {
				substitute(word, validWords,allWords);
			}
		};

		del.start();
		sub.start();
		ins.start();
		rev.start();

		try {
			sub.join();
			del.join();
			ins.join();
			rev.join();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		return validWords;
	}

	/**
	 * 
	 * Generate words within a single edit distance by single substitute
	 * @param word - word to be correct
	 * @param allWords - all word within a single edit distance (for next recursive call if needed)
	 * @param validWords - valid word within a single edit distance
	 */
	private void substitute(final String word, final Set<String> validWords,
			List<String> allWords) {
		for(int i=0; i < word.length(); ++i) {
			for(char c='a'; c <= 'z'; ++c) {
//				if(cMatrix.sub(word.charAt(i),c)>0) {
					String newstr = word.substring(0, i) + String.valueOf(c) +
							word.substring(i+1);
					allWords.add(newstr);
					isValidWord(validWords,newstr);
//				}
			}
		}
	}
	
	/**
	 * 
	 * Generate words by single insert
	 * @param word - word to be correct
	 * @param allWords - all word within a single edit distance (for next recursive call if needed)
	 * @param validWords - valid word within a single edit distance
	 */
	private void insert(final String word, final Set<String> validWords,
			List<String> allWords) {
		for(int i=0; i < word.length(); ++i) {
			for(char c='a'; c <= 'z'; ++c) {
//				if((i==0 && cMatrix.add('@',c)>0)||
//						(i!=0&&cMatrix.add(word.charAt(i-1),c)>0)) {
					String newstr = word.substring(0, i) + String.valueOf(c) + word.substring(i);
					allWords.add(newstr);
					isValidWord(validWords,newstr);
//				}
			}
		}
	}

	/**
	 * 
	 * Generate words by single reversal
	 * @param word - word to be correct
	 * @param allWords - all word within a single edit distance (for next recursive call if needed)
	 * @param validWords - valid word within a single edit distance
	 */
	private void reverse(final String word, final Set<String> validWords,
			List<String> allWords) {
		for(int i=0; i < word.length()-1; ++i){
//			if(cMatrix.rev(word.charAt(i),word.charAt(i+1))>0) {
				String newstr = word.substring(0, i) + word.substring(i+1, i+2) +
						word.substring(i, i+1) + word.substring(i+2);
				isValidWord(validWords,newstr);
				allWords.add(newstr);
//			}
		}
	}

	/**
	 * 
	 * Generate words by single delete
	 * @param word - word to be correct
	 * @param allWords - all word within a single edit distance (for next recursive call if needed)
	 * @param validWords - valid word within a single edit distance
	 */
	private void delete(final String word, final Set<String> validWords,
			List<String> allWords) {

		for(int i=0; i < word.length(); ++i){//delete i -th element
//			if((i==0 && cMatrix.del('@',word.charAt(i))>0)||
//					(i!=0 && cMatrix.del(word.charAt(i-1),word.charAt(i))>0)){   
				String newstr = word.substring(0, i) + word.substring(i+1);
				isValidWord(validWords,newstr);
				allWords.add(newstr);
//			}
		}
	}
	/**
	 * add  'word' to 'wordarray' if word is valid
	 * @param wordarray
	 * @param word
	 */
	private void isValidWord(final Set<String> wordarray,final String word) {
		if(trainedData.count(word)>0){
			wordarray.add(word);
		}
	}

	/**
	 * 
	 * @param word word to be corrected
	 * @return Map of Correct words and its score
	 */
	public Map<String,Double> getCorrect(final String word){

		double score;
		int editDistance = 1;
		
		List<String> allWords = Collections.synchronizedList(new ArrayList<String>());
		Map<String, Double> correct = new TreeMap<String, Double>();
		Set<String> validWords;
		
		
		validWords = edits(word,allWords);
		editDistance++;
		
		while(validWords.isEmpty() 
				&& editDistance <=MAX_EDIT 
				&& allWords.size()<=MAX_LOOPING){
			editDistance++;
			Iterator<String> iterator = allWords.iterator();
			List<String> newAllWords =
					Collections.synchronizedList(new ArrayList<String>());
			while(iterator.hasNext()) {
				String str = iterator.next();
				iterator.remove();
				validWords.addAll(edits(str, newAllWords));
			}
			allWords = newAllWords;
		}

		for (String str : validWords) {
			score = trainedData.unigramPrior(str);
			correct.put(str, score);
		}
		correct = sortByValue(correct);
		correct = normailze(correct);
		return correct;
	}
	
	/**
	 * Normalize scores
	 * @param correct
	 * @return
	 */
	private Map<String, Double> normailze(Map<String, Double> map) {
		
		double sum =0;
		for (Double values : map.values())
			sum+=values;
		for (Map.Entry<String, Double> entry : map.entrySet()) {
	          map.put(entry.getKey(), entry.getValue()/sum);
	    }
		return map;
	}

	/**
	 * For sorting of "Map <String,Integer>" according to  value
	 * @param map -input map
	 * @return sorted map
	 */
	
	static Map<String,Double> sortByValue(Map<String, Double> map) {
	     LinkedList<Entry<String,Double>> list =
	    		 new LinkedList<Entry<String,Double>>(map.entrySet());
	     Collections.sort(list, new Comparator<Entry<String,Double>>() {
			@Override
			public int compare(Entry<String,Double> arg0,
					Entry<String,Double> arg1) {
				// reverse Order
				return Double.compare(arg1.getValue(), arg0.getValue());
			}
	     });

	    Map<String, Double> result = new LinkedHashMap<String,Double>();
	    int cResult = 0;
	    for (Iterator<?> it = list.iterator(); it.hasNext();) {
	        @SuppressWarnings("unchecked")
			Entry<String,Double> entry = 
	        		(Entry<String, Double>) it.next();
	        result.put(entry.getKey(), entry.getValue());
	        if ( ++cResult > NO_OF_SUGGESTION ) break;
	    }
	    return result;
	} 
}
