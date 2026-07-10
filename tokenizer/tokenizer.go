package tokenizer

// Tokenizer is the common interface for all tokenizers in GoML.
// Both ByteTokenizer and BPETokenizer implement this.
type Tokenizer interface {
	Encode(text string) []int64
	Decode(tokens []int64) string
	VocabSize() int
}
