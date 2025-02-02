package main

import (
	"log"
	"os"
)

// $ go run ./...
// 何も返ってこなければ成功
func main() {
	sys := Mutex()
	model, err := KripkeModel(sys)
	if err != nil {
		log.Fatal(err)

	}
	model.WriteAsDot(os.Stdout)
}
