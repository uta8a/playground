package main

import (
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/mackee/go-readability"
)

func main() {
	// Fetch a web page
	resp, err := http.Get("https://blog.uta8a.net/post/2025-04-19-flux-first-step")
	if err != nil {
		log.Fatal(err)
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Fatal(err)
	}
	body := string(bodyBytes)
	options := readability.ReadabilityOptions{}
	// Parse and extract the main content
	article, err := readability.Extract(body, options)
	if err != nil {
		log.Fatal(err)
	}

	// Access the extracted content
	fmt.Println("Title:", article.Title)
	extracted := readability.ToMarkdown(article.Root)
	fmt.Println("---\n", extracted)
}
