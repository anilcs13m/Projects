package cli;

import java.io.BufferedInputStream;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import corpus.TrainedData;
import spellcheck.TrigramCheck;
import wordnet.Dictionary;

public class SentenceSpellCheck {
	static String inputFile = "sentences_input.tsv";
	static String outputFile = "sentences_output.tsv";

	public static void correctSentence(String infile,String outFile,
			Dictionary dict, TrainedData trainedData) {
		BufferedInputStream inBuffer;
		BufferedWriter outBuffer;

		System.err.println("Opening input file:" + infile);

		try {
			inBuffer = new BufferedInputStream(new FileInputStream(infile));
			outBuffer = new BufferedWriter(new FileWriter(outFile));
			StringBuilder line = new StringBuilder("");
			int c;
			char ch;
			while ((c = inBuffer.read()) != -1) {
				ch = (char) c;
				if (Character.isAlphabetic(ch))
					line.append(ch);
				else if(ch==','||ch=='.'||ch=='?'||ch=='!'){
					spellCheck(dict, trainedData,line.toString().trim(), outBuffer);
					line = new StringBuilder("");
					outBuffer.append(ch);
				} else {
					line.append(' ');
					if(ch == '\n') outBuffer.newLine();
				}
			}
			outBuffer.close();
			inBuffer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.err.println("Output File Saved:"+outFile);

	}

	private static void spellCheck(Dictionary dict, TrainedData trainedData,
			String curr, BufferedWriter buffer) throws IOException {

		TrigramCheck trigramCheck = new TrigramCheck(trainedData);

		StringBuilder trigrams = new StringBuilder();
		String next;
		List<String> words = new ArrayList<String>();

		for (String word : curr.split("\\s+")) {
			words.add(word);
		}

		for (String curr_word : words) {
			if (!trainedData.hasWord(curr_word.toLowerCase())
					&& !dict.hasWord(curr_word.toLowerCase())) {
				buffer.write(curr_word + "\t");

				trigrams.delete(0, trigrams.length());// make empty

				int index = words.indexOf(curr_word);
				for (int i = index - 2; i < index; i++) {
					if (i >= 0)
						trigrams.append(words.get(i) + " ");
				}
				if (index + 1 < words.size())
					next = words.get(index + 1);
				else
					next = "";

				Map<String, Double> map = trigramCheck.getCorrect(curr_word,
						trigrams.toString().trim(), next);

				buffer.write("[ ");
				for (String string : map.keySet()) {
					String score = String.format("%.2f", map.get(string) * 100);
					buffer.write(string + "  <" + score + ">  ");
				}
				buffer.write("] ");

			} else {
				buffer.write(curr_word + " ");
			}
		}
	}

	public static void main(String[] args) {

		Dictionary dict = new Dictionary();

		TrainedData trainedData = new TrainedData();

		correctSentence(inputFile, outputFile, dict, trainedData);

	}

}
