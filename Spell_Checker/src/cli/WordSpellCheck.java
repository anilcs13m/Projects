package cli;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import spellcheck.WordCheck;
import wordnet.Dictionary;
import corpus.TrainedData;

public class WordSpellCheck {

	private static final String inputFile = "words_input.tsv";
	private static final String ouputFile = "words_output.tsv";

	public static List<String> readWords(String file) {
		BufferedReader buffer;
		List<String> words = new ArrayList <String> ();

		try
		{
			buffer = new BufferedReader(new FileReader(file));   
			String line, temp[];
			System.err.println("Reading input file: "+file);
			while ((line = buffer.readLine())!= null){
				temp = line.split(" "); //split spaces

				words.add(temp[0]);
			}

			buffer.close();

		} catch(IOException e){
			System.err.println(e.toString());
		}
		return words;

	}

	public static void main(String[] args) {

		List<String> words = readWords(inputFile);
		Dictionary dict = new Dictionary();
		TrainedData trainedData = new TrainedData();

		BufferedWriter buffer;
		try {
			buffer = new BufferedWriter(new FileWriter(ouputFile));
			for (String str : words) {
				spellCheck(dict, trainedData, str,buffer);	
			}
			System.err.println("Saving output file:"+ouputFile);
			buffer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static void spellCheck(Dictionary dict, TrainedData trainedData,
			String word, BufferedWriter buffer) throws IOException {
		
		WordCheck wc = new WordCheck(trainedData);
		
	    if (!trainedData.hasWord(word.toLowerCase())
	    		&& !dict.hasWord(word.toLowerCase())) {
			buffer.write(word +"\t");
			Map<String, Double> map = wc.getCorrect(word);

			for (Map.Entry<String, Double> entry : map.entrySet()) {
				String newWord = entry.getKey();
				String score = String.format("%.2f",entry.getValue()*100);
				buffer.write(newWord+"  <" + score + ">\t");
		    }			

			buffer.newLine();
		} else {
			buffer.write(word +"  *");
			buffer.newLine();
		}
	}

}
