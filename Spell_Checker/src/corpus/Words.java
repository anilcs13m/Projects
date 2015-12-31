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
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import config.Properties;

class Words {
	
	private static final String FILE_NAME = "Data/WordsData.dat";
	private static final String BROWN_TXT_PATH = Properties.BROWN_PATH;
	private static final String CAT_FILE = BROWN_TXT_PATH + "cats.txt";
	
	private int vacabulary;
	private int count;
	
	Hashtable<String,Integer> wordsTable;
	
	
	public Words() {
		try {
			read(FILE_NAME);
			vacabulary = wordsTable.size();
		} catch (Exception e) {
			System.err.println("Data File Not Found: (" + e.getMessage() + ")");
			System.err.println("Reading taining file name from Text File..");

			wordsTable = new Hashtable<String, Integer>();
			count = 0;
			train(CAT_FILE);

			try {
				save(FILE_NAME);
			} catch (IOException e1) {
				e1.printStackTrace();
				System.exit(2);
			}
		}
	}
	
	@SuppressWarnings("unchecked")
	private void read(String file) throws IOException, ClassNotFoundException {
		InputStream buffer;
		buffer = new BufferedInputStream( new FileInputStream(file));
		ObjectInput input = new ObjectInputStream ( buffer );
		wordsTable = (Hashtable<String, Integer>) input.readObject();
		count = (Integer) input.readObject();
		input.close();
	}

	private void save(String file) throws IOException {
		OutputStream buffer;
		buffer = new BufferedOutputStream( new FileOutputStream(file));
		ObjectOutput output = new ObjectOutputStream ( buffer );
		output.writeObject(wordsTable);
		output.writeObject((Integer)count);
		output.close();
	}
	private void train(String catfile) {	
		List<String> file_names = new ArrayList<String>();
		BufferedReader buffer;
		try
		{
			buffer = new BufferedReader(new FileReader(catfile));   
			String line, temp[];

			while ((line = buffer.readLine())!= null)
			{ 
				temp = line.split(" "); //split spaces
				file_names.add(temp[0]);
			}
			buffer.close();
		} catch(Exception ex){
			ex.printStackTrace();
			System.exit(2);
		}

		BrownCorpusReader brownCorpusReader = new BrownCorpusReader();

		for (String filename : file_names) {

			try {
				System.err.println("Opening :" + filename +"..");
				buffer = new BufferedReader(new FileReader(BROWN_TXT_PATH+filename));
				count+= brownCorpusReader.getWords(buffer,wordsTable);
				buffer.close();
				
			} catch (IOException e) {
				e.printStackTrace();
				System.exit(2);
			}
		}
		vacabulary = wordsTable.size();
	}
	
	public int count(String word) {
		Integer ret = wordsTable.get(word);
		if(ret == null) return 0;
		return ret;
	}
	
	public double prior(String words) {
		return (count(words)+ 0.5)/(0.5*vacabulary + count);
	}
	
}
