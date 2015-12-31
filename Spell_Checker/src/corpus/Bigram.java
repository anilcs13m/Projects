package corpus;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.Hashtable;
import config.Properties;

class Bigram {

	private static final String FILE_NAME = "Data/BigramData.dat";
	private static final String BIGRAM_FILE = Properties.BIGRAM_FILE;

	Hashtable<String,Integer> bigramTable;
	Hashtable<String,Integer> unigramTable;
	private int vacabulary;

	public Bigram() {
		try {
			System.err.println("Init: Bigram..");
			read(FILE_NAME);
		} catch (Exception e) {
			System.err.println("Data File Not Found: (" + e.getMessage() + ")");
			System.err.println("Reading taining file name from Text File..");

			bigramTable = new Hashtable<String, Integer>();
			unigramTable = new Hashtable<String,Integer>();
			train();

			try {
				save(FILE_NAME);
			} catch (IOException e1) {
				e1.printStackTrace();
				System.exit(2);
			}
		}
		vacabulary = unigramTable.size();
	}


	@SuppressWarnings("unchecked")
	private void read(String file) throws IOException, ClassNotFoundException {
		InputStream buffer;
		System.err.println("Reading Data file:"+ file);
		buffer = new BufferedInputStream( new FileInputStream(file));
		ObjectInput input = new ObjectInputStream ( buffer );
		bigramTable = (Hashtable<String, Integer>) input.readObject();
		unigramTable = (Hashtable<String, Integer>) input.readObject();
		input.close();
		System.err.println("Finished");
	}

	private void save(String file) throws IOException {
		OutputStream buffer;
		buffer = new BufferedOutputStream( new FileOutputStream(file));
		ObjectOutput output = new ObjectOutputStream ( buffer );
		output.writeObject(bigramTable);
		output.writeObject(unigramTable);
		output.close();
	}
	private void train() {	
		BufferedReader reader;
		String [] bigram = new String [2];
		String unigram = new String ();
		int bigramCount =0 ,unigramCount = 0;
		
		try
		{

			System.err.println("Opening :" + BIGRAM_FILE +"..");
			reader = new BufferedReader(new FileReader(BIGRAM_FILE));
			
			String line;
			while ((line = reader.readLine()) != null) {
				line = line.trim();

				if (line.length() == 0)
					continue;

				String[] lineParts = line.split("\\s+");
				bigramCount = Integer.parseInt(lineParts[0]);

				bigram[0] = lineParts[1];
				bigram[1] = lineParts[2];

				if(bigram[0].equals(unigram)) {
					unigramCount+=bigramCount;
				} else {
					unigramTable.put(unigram, unigramCount);
					unigram = bigram[0];
					unigramCount = bigramCount;
				}
				bigramTable.put(bigram[0] + " "+ bigram[1],
						bigramCount);
			}
			unigramTable.put(unigram, unigramCount);

		} catch (IOException e) {
			e.printStackTrace();
			System.exit(2);
		}
	}

	public int count(String word) {
		Integer ret = bigramTable.get(word);
		if(ret == null) return 0;
		return ret;
	}
	private int unigramcount(String word) {
		Integer ret = unigramTable.get(word);
		if(ret == null) return 0;
		return ret;
	}

	public double prior(String word,String history) {
		return (count(history.trim() +" "+word.trim())+0.5)/
				(unigramcount(history.trim())+0.5*vacabulary);
	}
	
}
