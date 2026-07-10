package main

import (
	"fmt"
	"os"
	"time"

	"github.com/djeday123/goml/tokenizer"
)

func main() {
	fmt.Println("=== GoML BPE Tokenizer Test ===")

	// --- Load corpus ---
	corpus, err := loadCorpus("data/test_corpus.txt")
	if err != nil {
		fmt.Printf("Corpus file not found: %v\nUsing fallback...\n", err)
		corpus = fallbackCorpus()
	}
	fmt.Printf("Corpus: %d bytes\n\n", len(corpus))

	// --- Test 1: Train BPE (100 merges) ---
	fmt.Println("--- Test 1: Training ---")
	start := time.Now()
	tok := tokenizer.TrainBPE(corpus, 360) // 260 base + 100 merges
	fmt.Printf("Time: %v\n\n", time.Since(start))

	// --- Test 2: Multilingual roundtrip ---
	fmt.Println("--- Test 2: Multilingual Roundtrip ---")
	tests := []struct {
		lang, text string
	}{
		{"EN", "Artificial intelligence is transforming every industry"},
		{"EN", "Matrix multiplication must be cache-friendly"},
		{"RU", "Искусственный интеллект меняет мир"},
		{"RU", "Нейронные сети способны обрабатывать тексты"},
		{"AZ", "Süni intellekt hər bir sənayeni dəyişdirir"},
		{"AZ", "Azərbaycan əlifbası latın qrafikalı olub"},
		{"TR", "Yapay zeka her sektörü dönüştürüyor"},
		{"TR", "bilgisayarlaştıramadıklarımızdan"},
		{"MX", "Go + нейросети + süni intellekt = gələcək"},
		{"--", ""},
		{"--", "a"},
	}

	allOK := true
	for _, t := range tests {
		toks := tok.Encode(t.text)
		dec := tok.Decode(toks)
		ok := dec == t.text
		if !ok {
			allOK = false
		}
		sym := "✓"
		if !ok {
			sym = "✗"
		}
		ratio := float64(0)
		if len(toks) > 0 {
			ratio = float64(len(t.text)) / float64(len(toks))
		}
		fmt.Printf("  %s [%2s] %-45s → %3d tok (%.1fx)\n",
			sym, t.lang, trunc(t.text, 42), len(toks), ratio)
	}
	fmt.Printf("  Result: %s\n\n", passOrFail(allOK))

	// --- Test 3: Token details per language ---
	fmt.Println("--- Test 3: Token Details ---")
	for _, s := range []string{"neural network", "нейронная сеть", "neyron şəbəkə", "sinir ağları"} {
		toks := tok.Encode(s)
		fmt.Printf("  %q → %d tok: ", s, len(toks))
		for i, id := range toks {
			if i > 0 {
				fmt.Print(" | ")
			}
			fmt.Printf("%q", tok.DecodeToken(id))
		}
		fmt.Println()
	}
	fmt.Println()

	// --- Test 4: Special tokens ---
	fmt.Println("--- Test 4: Special Tokens ---")
	withBosEos := tok.EncodeWithSpecials("Salam dünya")
	fmt.Printf("  EncodeWithSpecials(%q) = %v\n", "Salam dünya", withBosEos)
	bosOK := withBosEos[0] == tokenizer.BosID
	eosOK := withBosEos[len(withBosEos)-1] == tokenizer.EosID
	fmt.Printf("  BOS=%d %s  EOS=%d %s\n\n", withBosEos[0], mark(bosOK), withBosEos[len(withBosEos)-1], mark(eosOK))

	// --- Test 5: Save / Load ---
	fmt.Println("--- Test 5: Save / Load ---")
	savePath := "/tmp/test_bpe_multi.merges"
	tok.Save(savePath)
	fmt.Printf("  Saved to %s\n", savePath)

	tok2, err := tokenizer.LoadBPE(savePath)
	if err != nil {
		fmt.Printf("  ✗ Load error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  Loaded: vocab=%d merges=%d\n", tok2.VocabSize(), tok2.NumMerges())

	loadOK := true
	for _, t := range tests {
		if !sliceEq(tok.Encode(t.text), tok2.Encode(t.text)) {
			loadOK = false
		}
	}
	fmt.Printf("  Loaded matches original: %s\n\n", mark(loadOK))

	// --- Test 6: Compression per language ---
	fmt.Println("--- Test 6: Compression ---")
	// Full corpus
	fullToks := tok.Encode(corpus)
	fmt.Printf("  Full:  %5d bytes → %4d tokens (%.2fx)\n",
		len(corpus), len(fullToks), float64(len(corpus))/float64(len(fullToks)))

	// Per-language segments (approximate splits by position)
	segments := []struct {
		lang       string
		start, end int
	}{
		{"EN", 0, 2500},
		{"RU", 2500, 5200},
		{"AZ", 5200, 8500},
		{"TR", 8500, 12000},
	}
	for _, seg := range segments {
		s, e := seg.start, seg.end
		if s >= len(corpus) {
			continue
		}
		if e > len(corpus) {
			e = len(corpus)
		}
		text := corpus[s:e]
		toks := tok.Encode(text)
		ratio := float64(len(text)) / float64(len(toks))
		fmt.Printf("  [%s]  %5d bytes → %4d tokens (%.2fx, %.1f bytes/tok)\n",
			seg.lang, len(text), len(toks), ratio, ratio)
	}
	fmt.Println()

	// --- Test 7: Larger vocab ---
	fmt.Println("--- Test 7: 200 Merges ---")
	start = time.Now()
	tok3 := tokenizer.TrainBPE(corpus, 460) // 260 + 200
	fmt.Printf("  Time: %v\n", time.Since(start))
	full3 := tok3.Encode(corpus)
	dec3 := tok3.Decode(full3)
	fmt.Printf("  %d bytes → %d tokens (%.2fx) roundtrip=%s\n\n",
		len(corpus), len(full3), float64(len(corpus))/float64(len(full3)), mark(dec3 == corpus))

	// --- Test 8: Interface ---
	fmt.Println("--- Test 8: Interface ---")
	var _ tokenizer.Tokenizer = tok
	var _ tokenizer.Tokenizer = tokenizer.NewByteTokenizer()
	fmt.Println("  ✓ BPETokenizer implements Tokenizer")
	fmt.Println("  ✓ ByteTokenizer implements Tokenizer")
	fmt.Println()

	fmt.Println("=== All tests complete ===")
}

// --- Helpers ---

func loadCorpus(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func fallbackCorpus() string {
	return `Artificial intelligence transforms industries worldwide.
Искусственный интеллект меняет мир быстрее чем любая технология.
Süni intellekt hər bir sənayeni dəyişdirir və yeni imkanlar yaradır.
Yapay zeka her sektörü dönüştürüyor ve yeni fırsatlar yaratıyor.
Machine learning, нейронные сети, maşın öyrənmə, makine öğrenimi.
Neural networks learn by gradient descent through backpropagation.
Градиентный спуск основной метод оптимизации нейронных сетей.
Qradiyent enişi neyron şəbəkələrinin əsas optimallaşdırma metodudur.
Gradyan iniş sinir ağlarının temel optimizasyon yöntemidir.`
}

func mark(ok bool) string {
	if ok {
		return "✓"
	}
	return "✗"
}

func passOrFail(ok bool) string {
	if ok {
		return "ALL PASSED ✓"
	}
	return "SOME FAILED ✗"
}

func trunc(s string, n int) string {
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}

func sliceEq(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}