package main

import (
	"bufio"
	"compress/bzip2"
	"encoding/xml"
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"
)

// ============================================================================
// wikiextract — Extract plain text from Wikipedia XML dumps
//
// Pure Go, no Python dependency. Reads .xml.bz2 dumps directly.
//
// Usage:
//   go run cmd/wikiextract/main.go -input trwiki-latest-pages-articles.xml.bz2 -output wiki_tr.txt -max-mb 400
//   go run cmd/wikiextract/main.go -input ruwiki-latest-pages-articles.xml.bz2 -output wiki_ru.txt -max-mb 400
// ============================================================================

// MediaWiki XML structure (simplified)
type Page struct {
	Title    string   `xml:"title"`
	NS       int      `xml:"ns"`
	Revision Revision `xml:"revision"`
}

type Revision struct {
	Text string `xml:"text"`
}

// Precompiled regexes for wikitext cleanup
var (
	// Templates: {{...}} (handles nested up to 3 levels)
	reTemplate = regexp.MustCompile(`\{\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\}`)

	// HTML tags
	reHTMLTag = regexp.MustCompile(`<[^>]+>`)

	// HTML comments
	reComment = regexp.MustCompile(`<!--[\s\S]*?-->`)

	// Tables: {| ... |}
	reTable = regexp.MustCompile(`(?s)\{\|.*?\|\}`)

	// Categories, files, images: [[Category:...]], [[File:...]]
	reCatFile = regexp.MustCompile(`\[\[(Category|Kategori|Kateqoriya|Категория|File|Image|Файл|Şəkil|Dosya):[^\]]*\]\]`)

	// Links: [[target|display]] → display, [[target]] → target
	reLink = regexp.MustCompile(`\[\[([^\]|]*\|)?([^\]]*)\]\]`)

	// External links: [http://... display] → display
	reExtLink  = regexp.MustCompile(`\[https?://[^\s\]]* ([^\]]*)\]`)
	reExtLink2 = regexp.MustCompile(`\[https?://[^\]]*\]`)

	// Bold/italic: '''bold''', ''italic''
	reBoldItalic = regexp.MustCompile(`'{2,3}`)

	// References: <ref>...</ref>, <ref ... />
	reRef = regexp.MustCompile(`(?s)<ref[^>]*>.*?</ref>|<ref[^/]*/\s*>`)

	// Headings: == Title == → Title
	reHeading = regexp.MustCompile(`(?m)^={2,6}\s*(.+?)\s*={2,6}\s*$`)

	// Multiple newlines → max 2
	reMultiNewline = regexp.MustCompile(`\n{3,}`)

	// Multiple spaces
	reMultiSpace = regexp.MustCompile(`[ \t]{2,}`)

	// Magic words: __TOC__, __NOTOC__, etc.
	reMagic = regexp.MustCompile(`__[A-Z]+__`)

	// Bullet/numbered lists: * item, # item, : indent, ; term
	reListMarker = regexp.MustCompile(`(?m)^[*#:;]+ *`)
)

func cleanWikitext(text string) string {
	// Skip redirects
	upper := strings.ToUpper(text)
	if strings.HasPrefix(upper, "#REDIRECT") ||
		strings.HasPrefix(upper, "#ПЕРЕНАПРАВЛЕНИЕ") ||
		strings.HasPrefix(upper, "#YÖNLENDİRME") ||
		strings.HasPrefix(upper, "#YÖNLENDIRME") ||
		strings.HasPrefix(upper, "#İSTİQAMƏTLƏNDİRMƏ") {
		return ""
	}

	s := text

	// Remove comments first
	s = reComment.ReplaceAllString(s, "")

	// Remove references
	s = reRef.ReplaceAllString(s, "")

	// Remove tables
	s = reTable.ReplaceAllString(s, "")

	// Remove templates (multiple passes for nesting)
	for i := 0; i < 5; i++ {
		prev := s
		s = reTemplate.ReplaceAllString(s, "")
		if s == prev {
			break
		}
	}

	// Remove categories, files, images
	s = reCatFile.ReplaceAllString(s, "")

	// Convert links
	s = reLink.ReplaceAllStringFunc(s, func(m string) string {
		inner := m[2 : len(m)-2] // strip [[ and ]]
		if idx := strings.LastIndex(inner, "|"); idx >= 0 {
			return inner[idx+1:]
		}
		return inner
	})

	// External links
	s = reExtLink.ReplaceAllString(s, "$1")
	s = reExtLink2.ReplaceAllString(s, "")

	// Remove HTML tags
	s = reHTMLTag.ReplaceAllString(s, "")

	// Bold/italic markers
	s = reBoldItalic.ReplaceAllString(s, "")

	// Headings → plain text with newlines
	s = reHeading.ReplaceAllString(s, "\n$1\n")

	// Magic words
	s = reMagic.ReplaceAllString(s, "")

	// List markers
	s = reListMarker.ReplaceAllString(s, "")

	// HTML entities
	s = strings.ReplaceAll(s, "&amp;", "&")
	s = strings.ReplaceAll(s, "&lt;", "<")
	s = strings.ReplaceAll(s, "&gt;", ">")
	s = strings.ReplaceAll(s, "&quot;", "\"")
	s = strings.ReplaceAll(s, "&nbsp;", " ")
	s = strings.ReplaceAll(s, "&ndash;", "–")
	s = strings.ReplaceAll(s, "&mdash;", "—")

	// Clean whitespace
	s = reMultiSpace.ReplaceAllString(s, " ")
	s = reMultiNewline.ReplaceAllString(s, "\n\n")
	s = strings.TrimSpace(s)

	return s
}

func main() {
	input := flag.String("input", "", "Input Wikipedia XML dump (.xml.bz2 or .xml)")
	output := flag.String("output", "", "Output text file")
	maxMB := flag.Int("max-mb", 300, "Stop after this many MB of text")
	minChars := flag.Int("min-chars", 200, "Skip articles shorter than this")
	verbose := flag.Bool("v", false, "Verbose (print every 1000th article)")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, `wikiextract — Extract text from Wikipedia XML dumps

Usage:
  go run cmd/wikiextract/main.go -input dump.xml.bz2 -output wiki.txt -max-mb 400

Examples:
  go run cmd/wikiextract/main.go -input data/trwiki-latest-pages-articles.xml.bz2 -output data/wiki_tr.txt -max-mb 400
  go run cmd/wikiextract/main.go -input data/ruwiki-latest-pages-articles.xml.bz2 -output data/wiki_ru.txt -max-mb 400
  go run cmd/wikiextract/main.go -input data/azwiki-latest-pages-articles.xml.bz2 -output data/wiki_az.txt -max-mb 300

Flags:
`)
		flag.PrintDefaults()
	}
	flag.Parse()

	if *input == "" || *output == "" {
		flag.Usage()
		os.Exit(1)
	}

	// Open input
	f, err := os.Open(*input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening %s: %v\n", *input, err)
		os.Exit(1)
	}
	defer f.Close()

	// Get file size for progress
	finfo, _ := f.Stat()
	totalSize := finfo.Size()

	var reader io.Reader
	if strings.HasSuffix(*input, ".bz2") {
		reader = bzip2.NewReader(f)
	} else {
		reader = f
	}

	// Buffered reader for performance
	bufReader := bufio.NewReaderSize(reader, 4*1024*1024) // 4MB buffer

	// Open output
	out, err := os.Create(*output)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating %s: %v\n", *output, err)
		os.Exit(1)
	}
	defer out.Close()
	writer := bufio.NewWriterSize(out, 1*1024*1024) // 1MB write buffer
	defer writer.Flush()

	// Parse XML
	decoder := xml.NewDecoder(bufReader)
	decoder.Strict = false
	decoder.CharsetReader = func(charset string, input io.Reader) (io.Reader, error) {
		return input, nil
	}

	targetBytes := int64(*maxMB) * 1024 * 1024
	totalBytes := int64(0)
	articles := 0
	skipped := 0
	startTime := time.Now()

	fmt.Printf("=== Wikipedia Text Extractor ===\n")
	fmt.Printf("Input:  %s (%.0f MB)\n", *input, float64(totalSize)/1048576)
	fmt.Printf("Output: %s\n", *output)
	fmt.Printf("Target: %d MB\n", *maxMB)
	fmt.Printf("Min article length: %d chars\n\n", *minChars)

	for {
		// Check target
		if totalBytes >= targetBytes {
			fmt.Printf("\nTarget reached: %.1f MB\n", float64(totalBytes)/1048576)
			break
		}

		// Read next token
		tok, err := decoder.Token()
		if err != nil {
			if err == io.EOF {
				fmt.Printf("\nEnd of dump reached\n")
			} else {
				// XML parse errors in wiki dumps are common, skip
				if *verbose {
					fmt.Fprintf(os.Stderr, "\nXML error (skipping): %v\n", err)
				}
			}
			break
		}

		// Look for <page> elements
		se, ok := tok.(xml.StartElement)
		if !ok || se.Name.Local != "page" {
			continue
		}

		// Decode the page
		var page Page
		if err := decoder.DecodeElement(&page, &se); err != nil {
			continue
		}

		// Skip non-article namespaces (ns=0 is articles)
		if page.NS != 0 {
			continue
		}

		// Skip empty
		if len(page.Revision.Text) == 0 {
			continue
		}

		// Clean wikitext
		cleaned := cleanWikitext(page.Revision.Text)

		// Skip short articles
		if utf8.RuneCountInString(cleaned) < *minChars {
			skipped++
			continue
		}

		// Write article
		n, _ := fmt.Fprintf(writer, "%s\n\n%s\n\n", page.Title, cleaned)
		totalBytes += int64(n)
		articles++

		// Progress
		if articles%1000 == 0 || *verbose && articles%100 == 0 {
			elapsed := time.Since(startTime).Seconds()
			mbDone := float64(totalBytes) / 1048576
			speed := mbDone / elapsed
			eta := (float64(*maxMB) - mbDone) / speed

			// Read position estimate (compressed position)
			pos, _ := f.Seek(0, io.SeekCurrent)
			pct := float64(pos) / float64(totalSize) * 100

			fmt.Printf("\r  %.1f / %d MB | %d articles | %.1f MB/s | dump: %.0f%% | ETA: %s     ",
				mbDone, *maxMB, articles, speed, pct, fmtDuration(eta))
		}
	}

	writer.Flush()

	// Summary
	elapsed := time.Since(startTime)
	fmt.Printf("\n\n=== Done ===\n")
	fmt.Printf("Output:   %s\n", *output)
	fmt.Printf("Size:     %.1f MB\n", float64(totalBytes)/1048576)
	fmt.Printf("Articles: %d (skipped %d short)\n", articles, skipped)
	fmt.Printf("Time:     %s\n", elapsed.Truncate(time.Second))
	if elapsed.Seconds() > 0 {
		fmt.Printf("Speed:    %.1f MB/s\n", float64(totalBytes)/1048576/elapsed.Seconds())
	}
}

func fmtDuration(seconds float64) string {
	if seconds < 0 || seconds > 86400 {
		return "?"
	}
	if seconds < 60 {
		return fmt.Sprintf("%ds", int(seconds))
	}
	if seconds < 3600 {
		return fmt.Sprintf("%dm%ds", int(seconds)/60, int(seconds)%60)
	}
	return fmt.Sprintf("%dh%dm", int(seconds)/3600, (int(seconds)%3600)/60)
}
