/**
 * 
 */
package config;

import java.io.FileInputStream;
import java.io.IOException;

public class  Properties{

	

	private static java.util.Properties prop;

	public static String WORDNET_DICT_PATH;
	public static String BROWN_PATH;
	public static String CMATIX_PATH;
	public static String TRIGRAM_FILE;
	public static String BIGRAM_FILE;

	static{

		prop = new java.util.Properties();
		try {
			//load a properties file
			prop.load(new FileInputStream("spellcheck.conf"));

			//get the property value and print it out
			WORDNET_DICT_PATH = prop.getProperty("WORDNET_DICT_PATH");
			BROWN_PATH = prop.getProperty("BROWN_PATH");
			CMATIX_PATH = prop.getProperty("CMATIX_PATH");
			TRIGRAM_FILE = prop.getProperty("TRIGRAM_FILE");
			BIGRAM_FILE = prop.getProperty("BIGRAM_FILE");
			//System.err.println(prop.toString());
		} catch (IOException ex) {
			ex.printStackTrace();
		}
	}

}
