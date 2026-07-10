package tokenizer

// ByteTokenizer is the simplest possible tokenizer — each byte is a token.
// Vocab size = 256. No subword merging.
// Good enough for training demos, will upgrade to BPE later.
type ByteTokenizer struct {
	vocabSize int
}

func NewByteTokenizer() *ByteTokenizer {
	return &ByteTokenizer{vocabSize: 256}
}

// Encode converts a string to token IDs (one per byte).
func (t *ByteTokenizer) Encode(text string) []int64 {
	bytes := []byte(text)
	tokens := make([]int64, len(bytes))
	for i, b := range bytes {
		tokens[i] = int64(b)
	}
	return tokens
}

// Decode converts token IDs back to a string.
func (t *ByteTokenizer) Decode(tokens []int64) string {
	bytes := make([]byte, len(tokens))
	for i, tok := range tokens {
		bytes[i] = byte(tok)
	}
	return string(bytes)
}

// DecodeToken converts a single token ID to string.
func (t *ByteTokenizer) DecodeToken(token int64) string {
	return string([]byte{byte(token)})
}

// VocabSize returns 256 (one token per byte value).
func (t *ByteTokenizer) VocabSize() int {
	return t.vocabSize
}
