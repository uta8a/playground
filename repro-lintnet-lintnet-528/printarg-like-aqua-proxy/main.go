// reference
// suzuki-shunsuke repro https://github.com/lintnet/lintnet/issues/528#issuecomment-2192696724
// aqua-proxy https://github.com/aquaproj/aqua-proxy/blob/9dc658ce3cb3c62203c1819d5828db5733b126c8/pkg/cli/xsys.go#L14-L28
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
