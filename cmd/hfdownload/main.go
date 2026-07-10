package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// ============================================================================
// hfdownload — Download text datasets from HuggingFace Hub
//
// Uses the HuggingFace datasets-server REST API to stream rows as JSON.
// No external Go dependencies, no Python, no parquet parsing.
//
// Authentication:
//   export HF_TOKEN=hf_xxxx   (get from huggingface.co/settings/tokens)
//   Without token: ~100 req/hr. With token: ~2000 req/hr.
//
// Examples:
//   go run cmd/hfdownload/main.go --preset wiki-az
//   go run cmd/hfdownload/main.go --preset wiki-tr --max-mb 500
//   go run cmd/hfdownload/main.go --preset oscar-ru --output oscar_ru.txt
//   go run cmd/hfdownload/main.go --dataset ccdv/arxiv-abstracts --field abstract
//   go run cmd/hfdownload/main.go --list --dataset wikimedia/wikipedia
// ============================================================================

const (
	rowsAPI   = "https://datasets-server.huggingface.co/rows"
	splitsAPI = "https://datasets-server.huggingface.co/splits"
	maxBatch  = 100
)

// Preset defines a known dataset configuration.
type Preset struct {
	Dataset string
	Config  string
	Split   string
	Field   string
	Output  string
	Desc    string
}

var presets = map[string]Preset{
	// Wikipedia (latest dump)
	"wiki-az": {"wikimedia/wikipedia", "20231101.az", "train", "text", "wiki_az.txt", "Azerbaijani Wikipedia"},
	"wiki-tr": {"wikimedia/wikipedia", "20231101.tr", "train", "text", "wiki_tr.txt", "Turkish Wikipedia"},
	"wiki-ru": {"wikimedia/wikipedia", "20231101.ru", "train", "text", "wiki_ru.txt", "Russian Wikipedia"},
	"wiki-en": {"wikimedia/wikipedia", "20231101.en", "train", "text", "wiki_en.txt", "English Wikipedia"},

	// OSCAR (Common Crawl, cleaned) — requires HF token + license acceptance
	"oscar-az": {"oscar-corpus/OSCAR-2301", "az", "train", "text", "oscar_az.txt", "OSCAR Azerbaijani"},
	"oscar-tr": {"oscar-corpus/OSCAR-2301", "tr", "train", "text", "oscar_tr.txt", "OSCAR Turkish"},
	"oscar-ru": {"oscar-corpus/OSCAR-2301", "ru", "train", "text", "oscar_ru.txt", "OSCAR Russian"},
	"oscar-en": {"oscar-corpus/OSCAR-2301", "en", "train", "text", "oscar_en.txt", "OSCAR English"},

	// Scientific
	"arxiv": {"ccdv/arxiv-abstracts", "default", "train", "abstract", "arxiv.txt", "arXiv abstracts"},

	// CC-100 (cleaned Common Crawl)
	"cc100-az": {"cc100", "az", "train", "text", "cc100_az.txt", "CC-100 Azerbaijani"},
	"cc100-tr": {"cc100", "tr", "train", "text", "cc100_tr.txt", "CC-100 Turkish"},
	"cc100-ru": {"cc100", "ru", "train", "text", "cc100_ru.txt", "CC-100 Russian"},
}

func main() {
	// --- Flags ---
	presetName := flag.String("preset", "", "Preset name (wiki-az, wiki-tr, oscar-ru, arxiv, etc.)")
	dataset := flag.String("dataset", "", "HuggingFace dataset ID (e.g. wikimedia/wikipedia)")
	config := flag.String("config", "", "Dataset config/subset (e.g. 20231101.az)")
	split := flag.String("split", "train", "Dataset split")
	field := flag.String("field", "text", "JSON field containing text")
	output := flag.String("output", "", "Output file path")
	maxMB := flag.Int("max-mb", 300, "Stop after this many MB")
	batch := flag.Int("batch", 100, "Rows per API request (max 100)")
	listConfigs := flag.Bool("list", false, "List available configs/presets")
	listPresets := flag.Bool("presets", false, "Show all preset names")
	clean := flag.Bool("clean", true, "Clean text (remove empty lines, trim)")
	verbose := flag.Bool("v", false, "Verbose output")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, `hfdownload — Download text from HuggingFace datasets

Usage:
  go run cmd/hfdownload/main.go [flags]

Presets (use --presets to see all):
  --preset wiki-az      Azerbaijani Wikipedia
  --preset wiki-tr      Turkish Wikipedia  
  --preset wiki-ru      Russian Wikipedia
  --preset wiki-en      English Wikipedia
  --preset oscar-az     OSCAR Azerbaijani (needs HF token)
  --preset arxiv        arXiv abstracts

Authentication:
  export HF_TOKEN=hf_xxxx    (from huggingface.co/settings/tokens)
  Required for OSCAR. Recommended for all (higher rate limits).

Flags:
`)
		flag.PrintDefaults()
	}
	flag.Parse()

	// --- List presets ---
	if *listPresets {
		fmt.Println("Available presets:")
		fmt.Println()
		for name, p := range presets {
			fmt.Printf("  %-12s  %-40s  %s\n", name, p.Dataset+"/"+p.Config, p.Desc)
		}
		fmt.Println()
		fmt.Println("Usage: go run cmd/hfdownload/main.go --preset wiki-az --max-mb 300")
		return
	}

	// --- Apply preset ---
	if *presetName != "" {
		p, ok := presets[*presetName]
		if !ok {
			fmt.Fprintf(os.Stderr, "Unknown preset: %s\nUse --presets to see available presets.\n", *presetName)
			os.Exit(1)
		}
		if *dataset == "" {
			*dataset = p.Dataset
		}
		if *config == "" {
			*config = p.Config
		}
		if *split == "train" && p.Split != "" {
			*split = p.Split
		}
		if *field == "text" && p.Field != "" {
			*field = p.Field
		}
		if *output == "" {
			*output = p.Output
		}
	}

	if *dataset == "" {
		fmt.Fprintln(os.Stderr, "Error: --dataset or --preset required")
		flag.Usage()
		os.Exit(1)
	}
	if *output == "" {
		*output = "output.txt"
	}
	if *batch > maxBatch {
		*batch = maxBatch
	}

	token := os.Getenv("HF_TOKEN")

	// --- List configs for dataset ---
	if *listConfigs {
		listDatasetConfigs(*dataset, token)
		return
	}

	// --- Download ---
	// Auto-discover config if not set
	if *config == "" {
		fmt.Print("Auto-discovering config... ")
		*config = autoDiscoverConfig(*dataset, token)
		fmt.Printf("found: %s\n", *config)
	}

	fmt.Println("=== HuggingFace Dataset Downloader ===")
	fmt.Printf("Dataset:  %s\n", *dataset)
	if *config != "" {
		fmt.Printf("Config:   %s\n", *config)
	}
	fmt.Printf("Split:    %s\n", *split)
	fmt.Printf("Field:    %s\n", *field)
	fmt.Printf("Output:   %s\n", *output)
	fmt.Printf("Max size: %d MB\n", *maxMB)
	if token != "" {
		fmt.Println("Auth:     HF_TOKEN set ✓")
	} else {
		fmt.Println("Auth:     no token (rate limited, set HF_TOKEN for faster downloads)")
	}
	fmt.Println()

	f, err := os.Create(*output)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating %s: %v\n", *output, err)
		os.Exit(1)
	}
	defer f.Close()

	client := &http.Client{Timeout: 30 * time.Second}

	targetBytes := int64(*maxMB) * 1024 * 1024
	totalBytes := int64(0)
	totalRows := int64(0)
	offset := 0
	errors := 0
	maxErrors := 10
	startTime := time.Now()

	for {
		// Check target
		if totalBytes >= targetBytes {
			fmt.Printf("\nTarget reached: %.1f MB\n", float64(totalBytes)/1048576)
			break
		}

		// Build URL
		url := fmt.Sprintf("%s?dataset=%s&split=%s&offset=%d&length=%d",
			rowsAPI, *dataset, *split, offset, *batch)
		if *config != "" {
			url += "&config=" + *config
		}

		if *verbose {
			fmt.Printf("  GET offset=%d\n", offset)
		}

		// Make request
		req, err := http.NewRequest("GET", url, nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error building request: %v\n", err)
			break
		}
		req.Header.Set("User-Agent", "GoML-HFDownload/1.0")
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}

		resp, err := client.Do(req)
		if err != nil {
			errors++
			fmt.Fprintf(os.Stderr, "\n  Request error (%d/%d): %v\n", errors, maxErrors, err)
			if errors >= maxErrors {
				fmt.Fprintln(os.Stderr, "Too many errors, stopping.")
				break
			}
			time.Sleep(time.Duration(errors*2) * time.Second)
			continue
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()

		// Handle rate limiting
		if resp.StatusCode == 429 {
			retryAfter := resp.Header.Get("Retry-After")
			wait := 60
			if retryAfter != "" {
				fmt.Sscanf(retryAfter, "%d", &wait)
			}
			fmt.Printf("\n  Rate limited. Waiting %ds...\n", wait)
			time.Sleep(time.Duration(wait) * time.Second)
			continue
		}

		if resp.StatusCode != 200 {
			errors++
			errMsg := string(body)
			if len(errMsg) > 200 {
				errMsg = errMsg[:200]
			}
			fmt.Fprintf(os.Stderr, "\n  HTTP %d (%d/%d): %s\n", resp.StatusCode, errors, maxErrors, errMsg)

			if resp.StatusCode == 401 || resp.StatusCode == 403 {
				fmt.Fprintln(os.Stderr, "  → Authentication required. Set HF_TOKEN environment variable.")
				fmt.Fprintln(os.Stderr, "  → Get token at: https://huggingface.co/settings/tokens")
				if strings.Contains(*dataset, "oscar") {
					fmt.Fprintln(os.Stderr, "  → OSCAR also requires accepting license at: https://huggingface.co/datasets/oscar-corpus/OSCAR-2301")
				}
				break
			}

			if errors >= maxErrors {
				fmt.Fprintln(os.Stderr, "Too many errors, stopping.")
				break
			}
			time.Sleep(time.Duration(errors*2) * time.Second)
			continue
		}

		if err != nil {
			errors++
			continue
		}
		errors = 0

		// Parse response
		var result RowsResponse
		if err := json.Unmarshal(body, &result); err != nil {
			fmt.Fprintf(os.Stderr, "\n  JSON parse error: %v\n", err)
			break
		}

		// No more rows
		if len(result.Rows) == 0 {
			fmt.Printf("\nDataset exhausted at offset %d\n", offset)
			break
		}

		// Extract text from rows
		batchBytes := int64(0)
		for _, row := range result.Rows {
			text, ok := row.Row[*field]
			if !ok {
				continue
			}

			str, ok := text.(string)
			if !ok {
				continue
			}

			// Clean
			if *clean {
				str = cleanText(str)
			}
			if len(str) == 0 {
				continue
			}

			n, _ := f.WriteString(str + "\n\n")
			batchBytes += int64(n)
			totalRows++
		}

		totalBytes += batchBytes
		offset += len(result.Rows)

		// Progress
		elapsed := time.Since(startTime).Seconds()
		mbDone := float64(totalBytes) / 1048576
		speed := mbDone / elapsed
		eta := float64(*maxMB-int(mbDone)) / speed

		fmt.Printf("\r  %.1f / %d MB | %d rows | %.2f MB/s | ETA: %s",
			mbDone, *maxMB, totalRows, speed, fmtDuration(eta))

		// Check if dataset is exhausted
		if result.NumRowsTotal > 0 && offset >= result.NumRowsTotal {
			fmt.Printf("\nDataset fully downloaded (%d rows)\n", result.NumRowsTotal)
			break
		}

		// Small sleep to be respectful
		time.Sleep(200 * time.Millisecond)
	}

	// Summary
	elapsed := time.Since(startTime)
	finalMB := float64(totalBytes) / 1048576
	fmt.Println()
	fmt.Println()
	fmt.Println("=== Done ===")
	fmt.Printf("File:     %s\n", *output)
	fmt.Printf("Size:     %.1f MB\n", finalMB)
	fmt.Printf("Rows:     %d\n", totalRows)
	fmt.Printf("Time:     %s\n", elapsed.Truncate(time.Second))
	if elapsed.Seconds() > 0 {
		fmt.Printf("Speed:    %.2f MB/s\n", finalMB/elapsed.Seconds())
	}
}

// ============================================================================
// API types
// ============================================================================

type RowsResponse struct {
	Rows         []RowEntry `json:"rows"`
	NumRowsTotal int        `json:"num_rows_total"`
}

type RowEntry struct {
	RowIdx int                    `json:"row_idx"`
	Row    map[string]interface{} `json:"row"`
}

type SplitsResponse struct {
	Splits []SplitEntry `json:"splits"`
}

type SplitEntry struct {
	Dataset string `json:"dataset"`
	Config  string `json:"config"`
	Split   string `json:"split"`
}

// ============================================================================
// Helpers
// ============================================================================

func listDatasetConfigs(dataset, token string) {
	url := fmt.Sprintf("%s?dataset=%s", splitsAPI, dataset)

	client := &http.Client{Timeout: 15 * time.Second}
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("User-Agent", "GoML-HFDownload/1.0")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := client.Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		fmt.Fprintf(os.Stderr, "HTTP %d: %s\n", resp.StatusCode, string(body))
		os.Exit(1)
	}

	var result SplitsResponse
	json.Unmarshal(body, &result)

	// Collect unique configs
	configs := make(map[string][]string)
	for _, s := range result.Splits {
		configs[s.Config] = append(configs[s.Config], s.Split)
	}

	fmt.Printf("Configs for %s (%d):\n\n", dataset, len(configs))
	for cfg, splits := range configs {
		fmt.Printf("  %-30s  splits: %s\n", cfg, strings.Join(splits, ", "))
	}
}

// autoDiscoverConfig queries /splits to find the first available config
func autoDiscoverConfig(dataset, token string) string {
	url := fmt.Sprintf("%s?dataset=%s", splitsAPI, dataset)

	client := &http.Client{Timeout: 15 * time.Second}
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("User-Agent", "GoML-HFDownload/1.0")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := client.Do(req)
	if err != nil {
		return "default"
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return "default"
	}

	var result SplitsResponse
	json.Unmarshal(body, &result)

	if len(result.Splits) > 0 {
		return result.Splits[0].Config
	}
	return "default"
}

func cleanText(s string) string {
	// Remove excessive whitespace
	lines := strings.Split(s, "\n")
	var cleaned []string
	emptyCount := 0
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			emptyCount++
			if emptyCount <= 1 {
				cleaned = append(cleaned, "")
			}
		} else {
			emptyCount = 0
			cleaned = append(cleaned, line)
		}
	}
	return strings.TrimSpace(strings.Join(cleaned, "\n"))
}

func fmtDuration(seconds float64) string {
	if seconds < 0 || seconds > 86400 {
		return "?"
	}
	d := time.Duration(seconds * float64(time.Second))
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	}
	if d < time.Hour {
		return fmt.Sprintf("%dm%ds", int(d.Minutes()), int(d.Seconds())%60)
	}
	return fmt.Sprintf("%dh%dm", int(d.Hours()), int(d.Minutes())%60)
}
