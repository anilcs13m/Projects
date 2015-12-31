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

class Trigram {

	private static final String FILE_NAME = "Data/TrigramData.dat";
	private static final String TRIGRAM_FILE = Properties.TRIGRAM_FILE;

	Hashtable<String,Integer> trigramTable;
	Hashtable<String,Integer> bigramTable;
	private int vacabulary;
	public Trigram() {
		System.err.println("Init: Trigram..");
		try {
			read(FILE_NAME);
		} catch (Exception e) {
			System.err.println("Data File Not Found: (" + e.getMessage() + ")");
			System.err.println("Reading taining file name from Text File..");

			trigramTable = new Hashtable<String, Integer>();
			bigramTable = new Hashtable<String, Integer>();

			train();

			try {
				save(FILE_NAME);
			} catch (IOException e1) {
				e1.printStackTrace();
				System.exit(2);
			}
		}
		vacabulary = bigramTable.size();
	}


	@SuppressWarnings("unchecked")
	private void read(String file) throws IOException, ClassNotFoundException {
		InputStream buffer;
		System.err.println("Reading Data file:"+ file);
		buffer = new BufferedInputStream( new FileInputStream(file));
		ObjectInput input = new ObjectInputStream ( buffer );
		trigramTable = (Hashtable<String, Integer>) input.readObject();
		bigramTable = (Hashtable<String, Integer>) input.readObject();
		input.close();
		System.err.println("Finished");
	}

	private void save(String file) throws IOException {
		OutputStream buffer;
		buffer = new BufferedOutputStream( new FileOutputStream(file));
		ObjectOutput output = new ObjectOutputStream ( buffer );
		output.writeObject(trigramTable);
		output.writeObject(bigramTable);
		output.close();
	}
	private void train() {	
		BufferedReader reader;
		String [] trigram = new String [3];
		String [] bigram = new String [2];
		int trigramCount =0 ,bigramCount = 0;

		try
		{

			System.err.println("Opening :" + TRIGRAM_FILE +"..");
			reader = new BufferedReader(new FileReader(TRIGRAM_FILE));
			String line;
			while ((line = reader.readLine()) != null) {
				line = line.trim();

				if (line.length() == 0)
					continue;

				String[] lineParts = line.split("\\s+");
				trigramCount = Integer.parseInt(lineParts[0]);

				trigram[0] = lineParts[1];
				trigram[1] = lineParts[2];
				trigram[2] = lineParts[3];

				if(trigram[0].equals(bigram[0]) && trigram[1].equals(bigram[1])) {
					bigramCount+=trigramCount;
				} else {
					bigramTable.put(bigram[0] + " "+ bigram[1], bigramCount);
					bigram[0] = trigram[0];
					bigram[1] = trigram[1];
					bigramCount = trigramCount;
				}
				trigramTable.put(trigram[0] + " "+ trigram[1]+ " "+ trigram[2],
						trigramCount);

			}
			reader.close();

		} catch (IOException e) {
			e.printStackTrace();
			System.exit(2);
		}
	}

	public int count(String word) {
		Integer ret = trigramTable.get(word);
		if(ret == null) return 0;
		return ret;
	}
	private int bigramcount(String word) {
		Integer ret = bigramTable.get(word);
		if(ret == null) return 0;
		return ret;
	}


	public double prior(String word,String history) {
		return (count(history.trim() +" "+word.trim())+0.5)/
				(bigramcount(history.trim())+vacabulary*0.5);
	}



}
