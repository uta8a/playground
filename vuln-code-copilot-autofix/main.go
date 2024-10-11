package main

import (
	"archive/zip"
	"os"
	"path/filepath"
	"strings"
)

func unzip(f string) {
	r, _ := zip.OpenReader(f)
	for _, f := range r.File {
		if !strings.Contains(f.Name, "..") {
			p, _ := filepath.Abs(f.Name)
			// GOOD: Check that path does not contain ".." before using it
			os.WriteFile(p, []byte("present"), 0666)
		}
	}
}
