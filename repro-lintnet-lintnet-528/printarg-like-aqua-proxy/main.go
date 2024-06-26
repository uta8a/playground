package main

import (
	"log"
	"os"

	"golang.org/x/sys/unix"
)

func main() {
	if err := core(os.Args...); err != nil {
		log.Fatal(err)
	}
}

const exePath = "../printargs/printarg"

func core(args ...string) error {
	return unix.Exec(exePath, append([]string{"aqua", "exec", "--", "echo"}, args[1:]...), os.Environ())
}
