package tokenizer

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// ============================================================================
// Token ID layout (shared across all GoML tokenizers):
//
//   0-255:   raw bytes (UTF-8 byte values)
//   256:     <pad>   padding
//   257:     <bos>   begin of sequence
//   258:     <eos>   end of sequence
//   259:     <unk>   unknown token
//   260+:    BPE merged tokens (ordered by merge priority)
//
// This layout ensures:
//   - ByteTokenizer output (0-255) is a valid subset of BPE output
//   - Merged vocab from LLaMA + Qwen can extend from 260+ with donor mapping
//   - Special tokens have fixed IDs across all configurations
// ============================================================================

const (
	NumBytes     = 256
	PadID        = int64(256)
	BosID        = int64(257)
	EosID        = int64(258)
	UnkID        = int64(259)
	FirstMergeID = int64(260)
)

// MergePair represents one BPE merge rule: tokens A and B merge into a new token.
// The resulting token ID is FirstMergeID + index_in_merges_list.
type MergePair struct {
	A, B int64
}

// BPETokenizer implements byte-level BPE (Sennrich et al., 2016).
//
// Training algorithm:
//  1. Start with vocab = 256 byte tokens + 4 special tokens
//  2. Count all adjacent token pairs in the corpus
//  3. Merge the most frequent pair into a new token
//  4. Repeat until target vocab size reached
//
// Encoding: apply learned merges in priority order (first learned = first applied).
// Decoding: look up byte sequence for each token ID, concatenate.
type BPETokenizer struct {
	merges    []MergePair      // ordered merge rules (index = priority)
	vocab     map[int64][]byte // id → byte sequence this token represents
	vocabSize int              // total vocab size (bytes + specials + merges)
}

// TrainBPE trains a byte-level BPE tokenizer on the given text corpus.
//
// Parameters:
//   - text: training corpus as a string
//   - targetVocabSize: desired total vocab size (260 + number of merges)
//
// Example: TrainBPE(corpus, 4356) creates 4096 merges (4356 - 260 base tokens).
func TrainBPE(text string, targetVocabSize int) *BPETokenizer {
	t := newBPEBase()

	// Convert entire corpus to byte token IDs
	raw := []byte(text)
	ids := make([]int64, len(raw))
	for i, b := range raw {
		ids[i] = int64(b)
	}

	numMerges := targetVocabSize - int(FirstMergeID)
	if numMerges <= 0 {
		return t
	}

	fmt.Printf("BPE training: %d bytes → %d merges\n", len(raw), numMerges)

	for m := 0; m < numMerges; m++ {
		// Count all adjacent pairs in current sequence
		counts := make(map[MergePair]int)
		for i := 0; i < len(ids)-1; i++ {
			p := MergePair{ids[i], ids[i+1]}
			counts[p]++
		}
		if len(counts) == 0 {
			fmt.Printf("  stopped at merge %d: no pairs left\n", m)
			break
		}

		// Find most frequent pair (deterministic tie-breaking by lower ID)
		var best MergePair
		bestCount := 0
		for p, c := range counts {
			if c > bestCount || (c == bestCount && (p.A < best.A || (p.A == best.A && p.B < best.B))) {
				best = p
				bestCount = c
			}
		}
		if bestCount < 2 {
			fmt.Printf("  stopped at merge %d: max pair freq = %d\n", m, bestCount)
			break
		}

		// Create new merged token
		newID := FirstMergeID + int64(m)
		merged := concatBytes(t.vocab[best.A], t.vocab[best.B])
		t.vocab[newID] = merged
		t.merges = append(t.merges, best)

		// Replace all occurrences of this pair in the corpus
		ids = replacePairInPlace(ids, best.A, best.B, newID)

		// Progress logging
		if (m+1)%500 == 0 || m < 5 || m == numMerges-1 {
			fmt.Printf("  merge %4d/%d  %q + %q → %q  freq=%-5d seq_len=%d\n",
				m+1, numMerges,
				safeStr(t.vocab[best.A]),
				safeStr(t.vocab[best.B]),
				safeStr(merged),
				bestCount, len(ids))
		}
	}

	t.vocabSize = int(FirstMergeID) + len(t.merges)

	// Final stats
	ratio := float64(len(raw)) / float64(len(ids))
	fmt.Printf("BPE done: vocab=%d  merges=%d  compression=%.2fx (%d bytes → %d tokens)\n",
		t.vocabSize, len(t.merges), ratio, len(raw), len(ids))

	return t
}

// ============================================================================
// Encode / Decode
// ============================================================================

// Encode converts text to a sequence of BPE token IDs.
//
// Algorithm: start with byte-level tokens, then apply each merge rule
// in training order (highest priority first). This guarantees the same
// segmentation that training produced.
//
// Complexity: O(len(text) × num_merges) worst case.
// The quick-check optimization skips merges not present in the current sequence.
// For production with 100K+ merges, use EncodeOptimized (priority queue, O(n log n)).
func (t *BPETokenizer) Encode(text string) []int64 {
	raw := []byte(text)
	if len(raw) == 0 {
		return nil
	}

	// Start with one token per byte
	ids := make([]int64, len(raw))
	for i, b := range raw {
		ids[i] = int64(b)
	}

	// Apply each merge in training order
	for i, merge := range t.merges {
		newID := FirstMergeID + int64(i)
		ids = replacePairInPlace(ids, merge.A, merge.B, newID)
	}

	return ids
}

// EncodeWithSpecials wraps text in <bos> ... <eos> tokens.
func (t *BPETokenizer) EncodeWithSpecials(text string) []int64 {
	tokens := t.Encode(text)
	result := make([]int64, 0, len(tokens)+2)
	result = append(result, BosID)
	result = append(result, tokens...)
	result = append(result, EosID)
	return result
}

// Decode converts BPE token IDs back to text.
// Unknown token IDs are silently skipped (no <unk> insertion).
func (t *BPETokenizer) Decode(tokens []int64) string {
	var buf []byte
	for _, id := range tokens {
		if b, ok := t.vocab[id]; ok {
			buf = append(buf, b...)
		}
	}
	return string(buf)
}

// DecodeToken converts a single token ID to its string representation.
func (t *BPETokenizer) DecodeToken(id int64) string {
	if b, ok := t.vocab[id]; ok {
		return string(b)
	}
	return "<unk>"
}

// VocabSize returns the total vocabulary size.
func (t *BPETokenizer) VocabSize() int {
	return t.vocabSize
}

// NumMerges returns the number of learned merge rules.
func (t *BPETokenizer) NumMerges() int {
	return len(t.merges)
}

// ============================================================================
// Save / Load
// ============================================================================

// Save writes the merge rules to a file.
//
// Format (one merge per line):
//
//	# GoML BPE v1
//	# vocab_size 4356
//	# num_merges 4096
//	101 32
//	116 104
//	...
//
// Each line contains two token IDs (A B) that merge into token FirstMergeID+line_index.
// This format is compact and allows rebuilding the full vocab from merges alone.
func (t *BPETokenizer) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	fmt.Fprintf(w, "# GoML BPE v1\n")
	fmt.Fprintf(w, "# vocab_size %d\n", t.vocabSize)
	fmt.Fprintf(w, "# num_merges %d\n", len(t.merges))
	for _, m := range t.merges {
		fmt.Fprintf(w, "%d %d\n", m.A, m.B)
	}
	return w.Flush()
}

// LoadBPE reads a merge file and reconstructs the BPE tokenizer.
// The vocab is rebuilt from merge rules: each merged token's bytes
// are the concatenation of its two component tokens' bytes.
func LoadBPE(path string) (*BPETokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	t := newBPEBase()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || line[0] == '#' {
			continue
		}
		var a, b int64
		if _, err := fmt.Sscanf(line, "%d %d", &a, &b); err != nil {
			continue
		}
		newID := FirstMergeID + int64(len(t.merges))
		t.vocab[newID] = concatBytes(t.vocab[a], t.vocab[b])
		t.merges = append(t.merges, MergePair{a, b})
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	t.vocabSize = int(FirstMergeID) + len(t.merges)
	return t, nil
}

// ============================================================================
// Donor Mapping (for future LLaMA / Qwen vocab merge)
// ============================================================================

// TokenBytes returns the raw byte sequence for a token ID.
// Used during weight inheritance to map donor embeddings to merged vocab.
func (t *BPETokenizer) TokenBytes(id int64) ([]byte, bool) {
	b, ok := t.vocab[id]
	return b, ok
}

// FindToken returns the token ID for a given byte sequence, or -1 if not found.
// Used to check if a donor token exists in the merged vocab.
func (t *BPETokenizer) FindToken(data []byte) int64 {
	s := string(data)
	for id, b := range t.vocab {
		if string(b) == s {
			return id
		}
	}
	return -1
}

// ============================================================================
// Internal helpers
// ============================================================================

// newBPEBase creates a BPETokenizer with the 260 base tokens (256 bytes + 4 specials).
func newBPEBase() *BPETokenizer {
	t := &BPETokenizer{
		vocab:     make(map[int64][]byte, 512),
		vocabSize: int(FirstMergeID),
	}
	// 256 byte tokens
	for i := 0; i < NumBytes; i++ {
		t.vocab[int64(i)] = []byte{byte(i)}
	}
	// 4 special tokens
	t.vocab[PadID] = []byte("<pad>")
	t.vocab[BosID] = []byte("<bos>")
	t.vocab[EosID] = []byte("<eos>")
	t.vocab[UnkID] = []byte("<unk>")
	return t
}

// replacePairInPlace scans ids and replaces all adjacent (a, b) with newID.
// Returns the original slice if the pair is not found (zero allocation).
func replacePairInPlace(ids []int64, a, b, newID int64) []int64 {
	// Quick scan: is this pair even present?
	found := false
	for i := 0; i < len(ids)-1; i++ {
		if ids[i] == a && ids[i+1] == b {
			found = true
			break
		}
	}
	if !found {
		return ids
	}

	// Build new sequence with replacements
	out := make([]int64, 0, len(ids))
	i := 0
	for i < len(ids) {
		if i+1 < len(ids) && ids[i] == a && ids[i+1] == b {
			out = append(out, newID)
			i += 2
		} else {
			out = append(out, ids[i])
			i++
		}
	}
	return out
}

// concatBytes concatenates two byte slices into a new slice.
func concatBytes(a, b []byte) []byte {
	c := make([]byte, len(a)+len(b))
	copy(c, a)
	copy(c[len(a):], b)
	return c
}

// safeStr returns a printable representation of bytes (escaping control chars).
func safeStr(b []byte) string {
	var sb strings.Builder
	sb.Grow(len(b) * 2)
	for _, c := range b {
		switch {
		case c == '\n':
			sb.WriteString(`\n`)
		case c == '\t':
			sb.WriteString(`\t`)
		case c == '\r':
			sb.WriteString(`\r`)
		case c >= 32 && c < 127:
			sb.WriteByte(c)
		default:
			fmt.Fprintf(&sb, `\x%02x`, c)
		}
	}
	return sb.String()
}
