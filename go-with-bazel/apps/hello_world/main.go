package main

import (
	"fmt"

	"github.com/google/uuid"
	"github.com/uta8a/playground/go-with-bazel/internal/reverse"
)

func main() {
	uuidStr := uuid.NewString()
	fmt.Printf("Hello, World!(%s)\n", uuidStr)
	fmt.Printf("Reversed: %s\n", reverse.String("Hello, World!"))
}
