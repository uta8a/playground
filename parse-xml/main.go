package main

import (
	"encoding/xml"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// Feed represents the top-level Atom feed
type Feed struct {
	XMLName   xml.Name  `xml:"feed"`
	Namespace string    `xml:"xmlns,attr"`
	Title     string    `xml:"title"`
	Link      []Link    `xml:"link"`
	ID        string    `xml:"id"`
	Updated   string    `xml:"updated"`
	Subtitle  string    `xml:"subtitle"`
	Generator Generator `xml:"generator"`
	Entries   []Entry   `xml:"entry"`
}

// Generator represents the generator of the feed
type Generator struct {
	URI  string `xml:"uri,attr"`
	Data string `xml:",chardata"`
}

// Link represents an Atom link element
type Link struct {
	Rel  string `xml:"rel,attr"`
	Type string `xml:"type,attr"`
	Href string `xml:"href,attr"`
}

// Entry represents an individual Atom entry (article)
type Entry struct {
	Title     string  `xml:"title"`
	Link      Link    `xml:"link"`
	ID        string  `xml:"id"`
	Published string  `xml:"published"`
	Updated   string  `xml:"updated"`
	Summary   string  `xml:"summary"`
	Author    Author  `xml:"author"`
	Content   Content `xml:"content"`
}

// Author represents the author information
type Author struct {
	Name string `xml:"name"` // Changed from "n" to "name" for standard Atom feeds
}

// Content represents the content of an entry
type Content struct {
	Type    string `xml:"type,attr"`
	XMLLang string `xml:"lang,attr"`
	XMLBase string `xml:"base,attr"`
	Data    string `xml:",innerxml"`
}

// RSSFeed represents the top-level RSS feed structure
type RSSFeed struct {
	XMLName xml.Name   `xml:"rss"`
	Version string     `xml:"version,attr"`
	Channel RSSChannel `xml:"channel"`
}

// RSSChannel represents the channel element in an RSS feed
type RSSChannel struct {
	Title         string    `xml:"title"`
	Link          string    `xml:"link"`
	Description   string    `xml:"description"`
	Language      string    `xml:"language"`
	LastBuildDate string    `xml:"lastBuildDate"`
	Generator     string    `xml:"generator"`
	Items         []RSSItem `xml:"item"`
}

// RSSItem represents an individual item in an RSS feed
type RSSItem struct {
	Title       string     `xml:"title"`
	Link        string     `xml:"link"`
	GUID        GUID       `xml:"guid"`
	Description string     `xml:"description"`
	PubDate     string     `xml:"pubDate"`
	Updated     string     `xml:"http://www.w3.org/2005/Atom updated"`              // Atom namespace
	Content     RSSContent `xml:"http://purl.org/rss/1.0/modules/content/ encoded"` // content namespace
}

// RSSContent represents the content:encoded element in RSS
type RSSContent struct {
	Data string `xml:",chardata"`
}

// GUID represents the guid element in an RSS item
type GUID struct {
	IsPermaLink string `xml:"isPermaLink,attr"`
	Value       string `xml:",chardata"`
}

func main() {
	// Check command line arguments
	if len(os.Args) < 2 {
		log.Fatalf("Usage: %s <xml-file>", os.Args[0])
	}

	filename := os.Args[1]

	// Open the XML file
	xmlFile, err := os.Open(filename)
	if err != nil {
		log.Fatalf("Error opening file: %v", err)
	}
	defer xmlFile.Close()

	// Read the file content
	xmlData, err := ioutil.ReadAll(xmlFile)
	if err != nil {
		log.Fatalf("Error reading file: %v", err)
	}

	// Determine feed type based on file extension
	extension := strings.ToLower(filepath.Ext(filename))

	switch extension {
	case ".xml": // Handle Atom feed
		parseAtomFeed(xmlData)
	case ".rss": // Handle RSS feed
		parseRSSFeed(xmlData)
	default:
		log.Fatalf("Unsupported file format: %s. Only .xml and .rss extensions are supported.", extension)
	}
}

// parseAtomFeed parses and displays Atom feed content
func parseAtomFeed(xmlData []byte) {
	// Create a Feed struct to parse into
	var feed Feed

	// Unmarshal XML data into the Feed struct
	err := xml.Unmarshal(xmlData, &feed)
	if err != nil {
		log.Fatalf("Error parsing XML: %v", err)
	}

	// Print basic feed information
	fmt.Println("=== Atom Feed ===")
	fmt.Printf("Feed Title: %s\n", feed.Title)
	fmt.Printf("Subtitle: %s\n", feed.Subtitle)
	fmt.Printf("Last Updated: %s\n", feed.Updated)
	fmt.Printf("Generator: %s", feed.Generator.Data)
	if feed.Generator.URI != "" {
		fmt.Printf(" (%s)", feed.Generator.URI)
	}
	fmt.Println()

	// Print information about each entry
	fmt.Printf("Found %d entries:\n\n", len(feed.Entries))
	for i, entry := range feed.Entries {
		fmt.Printf("Entry %d:\n", i+1)
		fmt.Printf("  Title: %s\n", entry.Title)
		if entry.Link.Href != "" {
			fmt.Printf("  Link: %s\n", entry.Link.Href)
		}
		if entry.Published != "" {
			fmt.Printf("  Published: %s\n", entry.Published)
		}
		if entry.Updated != "" {
			fmt.Printf("  Updated: %s\n", entry.Updated)
		}
		if entry.Author.Name != "" {
			fmt.Printf("  Author: %s\n", entry.Author.Name)
		}
		if entry.Summary != "" {
			fmt.Printf("  Summary: %s\n", entry.Summary)
		}
		fmt.Println()
	}
}

// parseRSSFeed parses and displays RSS feed content
func parseRSSFeed(xmlData []byte) {
	// Create an RSSFeed struct to parse into
	var rssFeed RSSFeed

	// Unmarshal XML data into the RSSFeed struct
	err := xml.Unmarshal(xmlData, &rssFeed)
	if err != nil {
		log.Fatalf("Error parsing RSS: %v", err)
	}

	// Print basic feed information
	fmt.Println("=== RSS Feed ===")
	fmt.Printf("Feed Title: %s\n", rssFeed.Channel.Title)
	fmt.Printf("Description: %s\n", rssFeed.Channel.Description)
	fmt.Printf("Last Build Date: %s\n", rssFeed.Channel.LastBuildDate)
	fmt.Printf("Generator: %s\n\n", rssFeed.Channel.Generator)

	// Print information about each item
	fmt.Printf("Found %d items:\n\n", len(rssFeed.Channel.Items))
	for i, item := range rssFeed.Channel.Items {
		fmt.Printf("Item %d:\n", i+1)
		fmt.Printf("  Title: %s\n", item.Title)
		fmt.Printf("  Link: %s\n", item.Link)
		fmt.Printf("  GUID: %s\n", item.GUID.Value)
		fmt.Printf("  Published: %s\n", item.PubDate)
		if item.Updated != "" {
			fmt.Printf("  Updated: %s\n", item.Updated)
		}
		fmt.Printf("  Description: %s\n", item.Description)
		if item.Content.Data != "" {
			contentPreview := item.Content.Data
			if len(contentPreview) > 100 {
				contentPreview = contentPreview[:100] + "..."
			}
			fmt.Printf("  Content: %s\n", contentPreview)
		}
		fmt.Println()
	}
}
