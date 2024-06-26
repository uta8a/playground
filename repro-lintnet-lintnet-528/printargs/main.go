package main

import (
	"fmt"
	"os"
)

func main() {
	args := os.Args
	for i, arg := range args {
		fmt.Printf("Argument %d: %s\n", i, arg)
	}
}
