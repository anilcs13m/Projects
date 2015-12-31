package corpus;

public class TaggedWord {
	private final String word;
	private final String tag;

	public TaggedWord(String word, String tag) {
		this.word = word;
		this.tag = tag;
	}

	@Override
	public boolean equals(Object o) {
		if (o == this)
			return true;

		if (!(o instanceof TaggedWord))
			return false;

		TaggedWord other = (TaggedWord) o;

		return other.word.equals(word) && other.tag.equals(tag);
	}

	public String tag() {
		return tag;
	}

	@Override
	public String toString() {
		return word + "/" + tag;
	}

	public String word() {
		return word;
	}
}
